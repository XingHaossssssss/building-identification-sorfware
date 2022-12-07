"""模型预测"""
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch
from rle import rle_encode
import argparse
from torchvision import transforms as T
import segmentation_models_pytorch as smp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # model = get_model()
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
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
            # score = model(image)['out'][0][0]
            score = model(image)[0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
    subm = pd.DataFrame(subm)
    subm.to_csv('./tmp.csv', index=None, header=None, sep='\t')

valid()
