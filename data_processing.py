"""数据处理"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from rle import rle_decode, rle_encode

import warnings
warnings.filterwarnings('ignore')

train_mask = pd.read_csv('./datasets/train_mask.csv', sep='\t', names=['name', 'mask'])
# 读取第一张图片，并将对应的rle解码为mask矩阵
img = cv2.imread('./datasets/train/' + train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])
# 判断rle解码和编码后是否相同
print(rle_encode(mask) == train_mask['mask'].iloc[0])
print('训练集大小:', train_mask.shape)

# 检查数据 count:每一行非空值数量 unique:不重复的离散值数目，去重之后的个数 top:最多的种类 freq:最多种类出现的频次
print(train_mask.isnull().any().describe())
print(train_mask.head())

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(mask)

NoBuild = 0
NoBuildIdx = []
BuildArea = 0

for idx in np.arange(train_mask.shape[0]):
    if pd.isnull(train_mask['mask'].iloc[idx]):
        NoBuild += 1
        NoBuildIdx.append(idx)
    else:
        mask = rle_decode(train_mask['mask'].iloc[idx])
        BuildArea += np.sum(mask)
# 统计训练集所有图片中建筑物区域平均大小
meanBuildArea = BuildArea / (train_mask.shape[0]-NoBuild)
print('训练集所有图片中建筑物区域的平均大小:%d' % meanBuildArea)
# 统计训练集所有图片中建筑物像素占所有像素的比例
BuildPixPerc = BuildArea / (train_mask.shape[0]*mask.shape[0]*mask.shape[1])
print('训练集所有图片中建筑物像素占所有像素比:%.3f' % BuildPixPerc)
# 统计所有图片中整图中没有任何建筑物像素占所有训练集图片的比例
NoBuildPerc = NoBuild / train_mask.shape[0]
print('训练集所有图片中无建筑物像素占:%.3f' % NoBuildPerc)

img = cv2.imread('./datasets/train/' + train_mask['name'].iloc[NoBuildIdx[1]])
plt.subplot(2, 2, 3)
plt.imshow(img)
img = cv2.imread('./datasets/train/' + train_mask['name'].iloc[NoBuildIdx[0]])
plt.subplot(2, 2, 4)
plt.imshow(img)
plt.show()



