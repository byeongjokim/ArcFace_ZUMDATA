from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *


def save_model(model, save_path, name, iter_cnt):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda")

    train_dataset = Dataset(opt.root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_zum_list(opt.zum_test_list)
    img_paths = [os.path.join(opt.root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    criterion = FocalLoss(gamma=2)
    # criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == "resnet18":
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == "resnet50":
        model = SEResNet_IR(50, mode='se_ir')
    else:
        exit

    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    
    print(model)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                                 lr=opt.lr, weight_decay=opt.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)

    start = time.time()
    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()

                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, iters, speed, loss.item(), acc))

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = zum_test(model, img_paths, opt.input_shape, identity_list, opt.zum_test_list, opt.test_batch_size)
