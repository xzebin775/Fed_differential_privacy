#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.distributions.binomial as binomial
from torch import maximum, minimum
from math import exp
from utils.options import args_parser
from utils.distributions import get_distributions
from utils.distributions import get_distributions2
from utils.perturbation import perturb_initial


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, round_num, w_C=None, w_R=None):

        args = args_parser()
        args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if args.method == 'initial' or round_num == 0:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        if args.communication == 'initial':
            for k in net.state_dict().keys():
                if net.state_dict()[k].shape == torch.Size([]):
                    continue

                # 构建indicator，指示矩阵
                if args.method == 'modified2':
                    indicator = get_distributions2(net.state_dict(), w_C, w_R, k)
                else:
                    indicator = get_distributions(net.state_dict(), w_C, w_R, k)

                # 对参数进行扰动
                perturb_initial(net.state_dict(), indicator, w_C, w_R, k)
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        elif args.communication == 'reduce':
            net_dict = copy.deepcopy(net.state_dict())
            for k in net.state_dict().keys():
                if net.state_dict()[k].shape == torch.Size([]):
                    continue
                # modified2情况下的reduce尚未完成
                if args.method == 'modified2':
                    indicator = get_distributions2(net.stat_dicat(), w_C, w_R, k)
                else:
                    indicator = get_distributions(net.state_dict(), w_C, w_R, k)
                net_dict[k] *= 0
                net_dict[k] += indicator
            return net_dict, sum(epoch_loss) / len(epoch_loss)
