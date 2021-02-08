from __future__ import print_function
import os
from data import Dataset, ZumDataset, ZUM
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

    train_dataset = ZUM(opt.train_root, opp.train_file_list, class_nums=opt.num_classes)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    print("{} train dataset".format(len(train_dataset)))

    eval_dataset = ZumDataset(opt.val_root, phase='eval', input_shape=opt.input_shape, classes=train_dataset.classes)
    evalloader = data.DataLoader(eval_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)
    print("{} eval dataset".format(len(eval_dataset)))

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = CBAMResNet(50, feature_dim=512, mode='ir')

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    print(model)
    model.to(device)
    metric_fc.to(device)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
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
                total_acc = 0
                total_num = 0
                for ii, data in enumerate(evalloader):
                    data_input, label = data
                    data_input = data_input.to(device)
                    label = label.to(device).long()
                    feature = model(data_input)
                    output = metric_fc(feature, label)

                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()

                    acc = np.mean((output == label).astype(int))

                    total_num += label.shape[0]
                    total_acc += acc * label.shape[0]

            print('[eval] epoch {} acc {}'.format(i, total_acc/total_num))
