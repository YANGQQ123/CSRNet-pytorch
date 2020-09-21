# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:52:02 2020

@author: young
"""


import os
import glob
import json
root = '/home/yqq/ShanghaiTech'
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train]

# 依次读取数据集中的每一张图片，将其放进列表img_paths
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
#print(img_paths)
with open('part_A_train.json','w') as f: json.dump(img_paths,f)