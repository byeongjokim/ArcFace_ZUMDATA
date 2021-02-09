from __future__ import print_function

import os
import time

import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

from data import ZUM
from models import *
from test import *

from config import Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def save_model(model, save_path, name, iter_cnt):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda")

    train_dataset = ZUM(opt.train_root, opt.train_list, class_nums=opt.num_classes)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  drop_last=False)
                                  
    identity_list = get_zum_list(opt.val_list)
    img_paths = [os.path.join(opt.val_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    
    criterion = FocalLoss(gamma=2)
    # criterion = torch.nn.CrossEntropyLoss()

    model = CBAMResNet(50, feature_dim=512, mode='ir')
    # model = resnet_face18(use_se=opt.use_se)
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    print(model)
    model.to(device)
    metric_fc.to(device)

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                                 lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

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

                print('[train] {} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)
        
        if i % opt.eval_interval == 0:
            with torch.no_grad():
                model.eval()
                acc = zum_test(model, img_paths, identity_list, opt.val_list, opt.val_batch_size)

            print('[eval] epoch {} acc {}'.format(i, acc))
