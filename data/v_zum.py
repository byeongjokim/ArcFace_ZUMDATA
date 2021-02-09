#!/usr/bin/env python
# encoding: utf-8
'''
@author: byeongjokim
@contact: 
@file: zum.py
@time: 2021/02/08 19:09
@desc: ZUM dataset loader
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = cv2.resize(img, (112, 112))
            return img
    except IOError:
        print('Cannot load image ' + path)


class V_ZUM(data.Dataset):
    def __init__(self, root, file_list, loader=img_loader):

        self.root = root

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

        self.loader = loader

        self.left = []
        self.right = []
        self.flag = []

        with open(file_list) as f:
            pairs = f.read().splitlines()

        for i, p in enumerate(pairs):
            # p -> img1 img2 1 (Same)
            # p -> img1 img2 0 (Diff)
            p = p.split(" ")

            self.left.append(p[0])
            self.right.append(p[1])
            self.flag = int(p[2])
        
        print("dataset size: ", len(self.left))

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.left[index]))
        img_r = self.loader(os.path.join(self.root, self.right[index]))

        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]
        flag = self.flag[index]

        if self.transform is not None:
            imglist = [self.transform(img) for img in imglist]
        
        return imglist, flag

    def __len__(self):
        return len(self.left)
