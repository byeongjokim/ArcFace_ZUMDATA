# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
import argparse
from torch.nn import DataParallel
from torchvision import transforms as T
from PIL import Image, ImageOps


def get_zum_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list

def load_image(img_path, input_shape):
    normalize = T.Normalize(mean=[0.5], std=[0.5])
    transforms = T.Compose([
        T.CenterCrop(input_shape[1:]),
        T.ToTensor(),
        normalize
    ])
    data = Image.open(img_path)
    
    if input_shape[0] == 1:
        data = data.convert('L')

    data_f = ImageOps.mirror(data)

    data = np.asarray(transforms(data))
    data_f = np.asarray(transforms(data_f))

    image = np.stack((data, data_f), 0)
    return image


def get_featurs(model, test_list, input_shape, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path, input_shape)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def zum_test(model, img_paths, input_shape, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, input_shape, batch_size=batch_size)
    t = time.time() - s
    print('[{}] total time is {}, average time is {}'.format(time.strftime('%c', time.localtime(time.time())), t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print("[{}]".format(time.strftime('%c', time.localtime(time.time()))), 'zum face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--num_classes', default=200, type=int)
    parser.add_argument('--metric', default="arc_margin")
    parser.add_argument('--easy_margin', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    parser.add_argument('--optimizer', default="sgd")
    
    parser.add_argument('--inf_root', default="/")
    parser.add_argument('--inf_json', default="/home/kbj/projects/ZUMDATA/inf/people/OBS")
    
    parser.add_argument('--root', default="/home/kbj/projects/ZUMDATA/total")
    
    parser.add_argument('--total_list', default="/home/kbj/projects/ZUMDATA/real_total.txt")
    parser.add_argument('--train_list', default="/home/kbj/projects/ZUMDATA/real_train.txt")
    parser.add_argument('--val_list', default="/home/kbj/projects/ZUMDATA/real_val.txt")
    parser.add_argument('--test_list', default="/home/kbj/projects/ZUMDATA/real_test.txt")
    parser.add_argument('--zum_test_list', default="/home/kbj/projects/ZUMDATA/real_pair.txt")
    
    parser.add_argument('--checkpoints_path', default="checkpoints")
    
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--lr_step', default=10, type=int)
    parser.add_argument('--lr_decay', default=0.2, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu_id', default="1")
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    
    opt = parser.parse_args()
    
    if opt.backbone == "resnet18":
        opt.input_shape = (1, 128, 128)
        opt.test_model_path = 'checkpoints/resnet18/resnet18_30.pth'
    else:
        opt.input_shape = (3, 112, 112)
        opt.test_model_path = 'checkpoints/resnet50/resnet50_50.pth'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    
    if opt.backbone == "resnet18":
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == "resnet50":
        model = SEResNet_IR(50, mode='se_ir')
    else:
        exit

    model = DataParallel(model)

    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    identity_list = get_zum_list(opt.zum_test_list)
    img_paths = [os.path.join(opt.root, each) for each in identity_list]

    model.eval()
    zum_test(model, img_paths, opt.input_shape, identity_list, opt.zum_test_list, opt.test_batch_size)




