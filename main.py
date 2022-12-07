"""模型训练和预测"""
import numpy as np
import pandas as pd
import os
import cv2
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import albumentations as A
import torch
import torch.utils.data as D
from rle import rle_encode, rle_decode
from Tianchidataset import TianChiDataset
from loss import loss_fn
import argparse
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model():
    # model = torchvision.models.segmentation.fcn_resnet101(True)
    # model = torchvision.models.segmentation.fcn_resnet50(True)
    """
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    """
    """
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    # model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # 使用FCN时使用
    return model


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        # output = model(image)['out']  # FCN使用
        output = model(image)  # Unet使用
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic segmentation network')
    parser.add_argument('--modelDir',
                        help='saved model path name',
                        default="./checkpoints/model_best.pth",
                        type=str)
    parser.add_argument('--data_path',
                        help='dataset path',
                        default='./datasets',
                        type=str)
    parser.add_argument('--epoch',
                        help='total train epoch num',
                        default=20,
                        type=int)
    parser.add_argument('--batch_size',
                        help='total train epoch num',
                        default=8,
                        type=int)
    parser.add_argument('--image_size',
                        help='total train epoch num',
                        default=256,
                        type=int)
    parser.add_argument('--gpu_ids',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
                        default=[0],
                        type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    #  --------------------------加载数据及数据增强----------------------------
    train_mask = pd.read_csv(os.path.join(args.data_path, 'train_mask.csv'), sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: os.path.join(args.data_path,'train/') + x)
    mask = rle_decode(train_mask['mask'].iloc[0])
    print(rle_encode(mask) == train_mask['mask'].iloc[0])
    # print(train_mask.shape)
    trfm = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                          saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
    ])
    dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        trfm, False
    )
    # print(TianChiDataset.__getitem__(dataset, 1))
    valid_idx, train_idx = [], []
    for i in range(len(dataset)):
        '''
        if i % 7 == 0:
            valid_idx.append(i)
        #     else:
        elif i % 7 == 1:
            train_idx.append(i)
        '''
        valid_idx.append(i)
        train_idx.append(i)

    train_ds = D.Subset(dataset, train_idx)
    valid_ds = D.Subset(dataset, valid_idx)
    # 定义训练集和验证集数据加载器
    loader = D.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vloader = D.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    '''
    print(len(loader))
    for data in loader:
        id, img = data
        print(len(data))
        print(id.shape)
        print(img.shape)
    '''
    # ----------------------------加载模型及优化器------------------------------------
    model = get_model()
    model.to(DEVICE)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids, output_device=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    train_loss = []
    if os.path.exists(args.modelDir):
        checkpoint = torch.load(args.modelDir)
        model.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'train_loss' in checkpoint:
            train_loss = checkpoint['train_loss']
        print("load model from {}".format(args.modelDir))
    else:
        start_epoch = 0
        print("==> no checkpoint found at '{}'".format(args.modelDir))

    # ----------------------------训练-----------------------------------
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)
    best_loss = 10

    for epoch in range(start_epoch, args.epoch):
        losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm(loader):
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()
            # output = model(image)['out'] # 使用FCN时使用
            output = model(image)  # 使用Unet时使用
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())

        vloss = validation(model, vloader, loss_fn)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss, (time.time() - start_time) / 60 ** 1))
        train_loss.append(np.array(losses).mean())
        if vloss < best_loss:
            best_loss = vloss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss

            }
            torch.save(state, args.modelDir)  # 保存整个模型

    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(train_loss, label="loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# --------------------------------验证-----------------------------------
def valid():
    args = parse_args()
    trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(args.image_size),
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688],
                    [0.131, 0.177, 0.101]),
    ])
    subm = []
    model = get_model()
    model.to(DEVICE)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids, output_device=0)
    if os.path.exists(args.modelDir):
        checkpoint = torch.load(args.modelDir)
        model.load_state_dict(checkpoint['state_dict'])
        print("load model from {}".format(args.modelDir))
    model.eval()
    test_mask = pd.read_csv(os.path.join(args.data_path, 'test_a_samplesubmit.csv'), sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: os.path.join(args.data_path, 'test_a/') + x)

    for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
        image = cv2.imread(name)
        image = trfm(image)
        with torch.no_grad():
            image = image.to(DEVICE)[None]
            # score = model(image)['out'][0][0]  # FCN
            score = model(image)[0][0]  # Unet
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
    subm = pd.DataFrame(subm)
    subm.to_csv('./tmp.csv', index=None, header=None, sep='\t')


if __name__ == '__main__':
    main()
    valid()