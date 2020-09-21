# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:14:57 2020

@author: young
"""
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import scipy.spatial
import json
from matplotlib import cm as CM

from model import CSRNet
import torch


root = 'D:\GitHub\projects\ShanghaiTech_Crowd_Counting_Dataset'
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]

# 依次读取数据集中的每一张图片，将其放进列表img_paths
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
    print('图片数量：', len(img_paths))

def gaussian_filter_density(gt):
    print(gt.shape)

    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    #构造KDTree寻找相邻的人头位置
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            #相邻三个人头的平均距离，其中beta=0.3
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


#读取图片
for img_path in img_paths:
    print(img_path)
    # 获取每张图片对应的mat标记文件
    mat = io.loadmat(img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat'))
    img = plt.imread(img_path)
    # 生成密度图
    gt_density_map = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            gt_density_map[int(gt[i][1]), int(gt[i][0])] = 1
    gt_density_map = gaussian_filter_density(gt_density_map)
    # 保存生成的密度图
    with h5py.File(img_path.replace('images', 'ground_truth').replace('.jpg', '.h5'), 'w') as hf:
        hf['density'] = gt_density_map

        #测试
    print('总数量=',len(gt))
    print('密度图=',gt_density_map.sum())
    

