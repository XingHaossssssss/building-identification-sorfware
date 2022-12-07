"""测试集预测结果图像绘制"""
import numpy as np

from rle import rle_decode
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def plt_1x2(img1, img2, name):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.suptitle(name)
    plt.show()


def loadtestdata(name, idx):
    result = pd.read_csv('./submit/' + name, sep='\t', names=['name', 'mask'])
    img = cv2.imread('./datasets/test_a/' + result['name'].iloc[idx])
    if result['mask'].iloc[idx] is np.NAN:
        mask = cv2.imread('./datasets/test_a/' + result['name'].iloc[idx])
    else:
        mask = rle_decode(result['mask'].iloc[idx])
    return img, mask

# 读取图片
idx = 450

test_img, fcn_resnet101_15_mask = loadtestdata('fcn_resnet101_15_tmp.csv', idx)
# plt_1x2(test_img, fcn_resnet101_15_mask, 'fcn_resnet101_15')

test_img, fcn_resnet101_10_mask = loadtestdata('fcn_resnet101_10_tmp.csv', idx)
# plt_1x2(test_img, fcn_resnet101_10_mask, 'fcn_resnet101_10')

test_img, fcn_resnet50_15_mask = loadtestdata('fcn_resnet50_15_tmp.csv', idx)
# plt_1x2(test_img, fcn_resnet50_15_mask, 'fcn_resnet50_15')

test_img, Unet_resnet50_15_mask = loadtestdata('Unet_resnet50_15_tmp.csv', idx)
# plt_1x2(test_img, Unet_resnet50_15_mask, 'Unet_resnet50_15')

test_img, Unet_efficientB4_15_mask = loadtestdata('Unet_efficientB4_15_tmp.csv', idx)
# plt_1x2(test_img, Unet_efficientB4_15_mask, 'Unet_efficientB4_15')

test_img, UnetPP_efficientB4_15_mask = loadtestdata('Unet++_efficientB4_15_tmp.csv', idx)
# plt_1x2(test_img, UnetPP_efficientB4_15_mask, 'Unet++_efficientB4_15')

test_img, UnetPP_efficientB4_20_mask = loadtestdata('Unet++_efficientB4_20_ChangeLossandData_tmp.csv', idx)
plt_1x2(test_img, UnetPP_efficientB4_15_mask, 'Unet++_efficientB4_15')


def plt_3x3():
    plt.figure()
    plt.subplot(3, 3, 2)
    plt.imshow(test_img)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(3, 3, 4)
    plt.imshow(fcn_resnet101_15_mask)
    plt.title('fcn_resnet101_15')
    plt.axis('off')
    plt.subplot(3, 3, 5)
    plt.imshow(fcn_resnet101_10_mask)
    plt.title('fcn_resnet101_10')
    plt.axis('off')
    plt.subplot(3, 3, 6)
    plt.imshow(fcn_resnet50_15_mask)
    plt.title('fcn_resnet50_15')
    plt.axis('off')
    plt.subplot(3, 3, 7)
    plt.imshow(Unet_resnet50_15_mask)
    plt.title('Unet_resnet_15')
    plt.axis('off')
    plt.subplot(3, 3, 8)
    plt.imshow(Unet_efficientB4_15_mask)
    plt.title('Unet_efficientB4_15')
    plt.axis('off')
    plt.subplot(3, 3, 9)
    plt.imshow(UnetPP_efficientB4_15_mask)
    plt.title('Unet++_efficientB4_15')
    plt.axis('off')
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.35)
    plt.show()


# plt_3x3()