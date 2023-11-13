#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import re

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update_duchi import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed_duchi import FedAvg
from models.test import test_img
from utils.initCR import init_cr
import torchvision.models as models
import torch.nn as nn

CIFAR_PATH="../data/cifar100"
mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers=2

def cifar100_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar100_training = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True,
                                                      transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=100, shuffle=True,
                                              num_workers=args.num_workers)

    cifar100_testing = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True,
                                                     transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args.iid)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

        # if args.iid:
        #     dict_users = cifar_iid(dataset_train, args.num_users)
        # else:
        #     exit('Error: only consider IID setting in CIFAR10')
        # 不管args.iid的值，始终认为args.iid为真
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = models.resnet18(pretrained=True)
        fc_features = net_glob.fc.in_features
        net_glob.fc = nn.Linear(fc_features, 10)
        net_glob.to(args.device)
    # elif args.model=='resnet' and args.dataset=='imgnet':
    #     net_glob = models.resnet18(pretrained=True)
    #     fc_features = net_glob.fc.in_features
    #     net_glob.fc = nn.Linear(fc_features, 10)
    #     net_glob.to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    w_C, w_R = init_cr(net_glob, False)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    w_global = copy.deepcopy(w_glob)
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # print("user{} is running".format(idx))
            if iter >= 1:
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device), iter, w_C, w_R)
            else:
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device), iter)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # if iter == 0:
        #     w_glob, w_C, w_R = FedAvg(w_locals, w_C, w_R, True)
        #     # w_glob=FedAvg(w_locals,False)
        # else:
        #     w_glob, w_C, w_R = FedAvg(w_locals, w_C, w_R, False)
        w_glob, w_C, w_R = FedAvg(w_locals, w_C, w_R, iter)
        # w_glob=FedAvg(w_locals,False)
        w_global = copy.deepcopy(w_glob)
        # print(w_C['conv1.weight'][0][0][0][0])
        # print(w_R['conv1.weight'][0][0][0][0])
        # print(w_R['conv1.weight'][0][0][0][1])
        # print(w_glob['conv1.weight'][0][0][0][0])

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}+model_{}+epochs_{}+users_{}+frac_{}_+iid{}+method_{}+com_{}+epslon_{}.png'.
                format(args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid, args.method,
                       args.communication, args.epslon))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
