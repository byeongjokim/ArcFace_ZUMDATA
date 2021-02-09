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


class ZUM(data.Dataset):
    def __init__(self, root, file_list, loader=img_loader, class_nums=None):

        self.root = root

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

        self.loader = loader

        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        
        if not class_nums:
            self.class_nums = len(np.unique(self.label_list))
        else:
            self.class_nums = class_nums
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))

        # random flip with ratio of 0.5
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return len(self.image_list)
