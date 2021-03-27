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
import argparse
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
#     opt = Config()
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
    
    parser.add_argument('--total_list', default="/home/kbj/projects/ZUMDATA/total.txt")
    parser.add_argument('--train_list', default="/home/kbj/projects/ZUMDATA/train.txt")
    parser.add_argument('--val_list', default="/home/kbj/projects/ZUMDATA/val.txt")
    parser.add_argument('--test_list', default="/home/kbj/projects/ZUMDATA/test.txt")
    parser.add_argument('--zum_test_list', default="/home/kbj/projects/ZUMDATA/pair.txt")
    
    parser.add_argument('--checkpoints_path', default="checkpoints")
    
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--lr_step', default=10, type=int)
    parser.add_argument('--lr_decay', default=0.2, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    
    parser.add_argument('--use_gpu', action='store_true', default=True)
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
    
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda")

    train_dataset = Dataset(opt.root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_zum_list(opt.zum_test_list)
    img_paths = [os.path.join(opt.root, each) for each in identity_list]

    print('[{}] {} train iters per epoch:'.format(time.strftime('%c', time.localtime(time.time())), len(trainloader)))

    criterion = FocalLoss(gamma=2)
    # criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == "resnet18":
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == "resnet50":
        model = SEResNet_IR(50, mode='se_ir')
    else:
        exit

    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    print("[{}]".format(time.strftime('%c', time.localtime(time.time()))))
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
                print('[{}] {} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time.strftime('%c', time.localtime(time.time())), time_str, i, iters, speed, loss.item(), acc))

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = zum_test(model, img_paths, opt.input_shape, identity_list, opt.zum_test_list, opt.test_batch_size)
