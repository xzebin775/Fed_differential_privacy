#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.distributions.binomial as binomial
from math import exp
from utils.options import args_parser
import numpy as np
import random
from sklearn import metrics



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

    def train(self, net,need,w_C=None,w_R=None):
    #def train(self,net,need):
        args=args_parser()
        args.device=torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu!=-1 else 'cpu')

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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        singleUserIndicator=copy.deepcopy(net.state_dict())
        if need:
            epslon = 3.0
            for k in net.state_dict().keys():
                 R=w_R[k]
                 C=w_C[k]
                # if k=='conv1.weight':
                   # print(R[0][0][0][0])
                 distributions=copy.deepcopy(net.state_dict()[k])
                 distributions=torch.div((distributions-C),R)*((exp(epslon)-1)/(2*exp(epslon)+2))+0.5
                 m=binomial.Binomial(1,distributions)
                 indicator=m.sample()
                 # if k=='conv1.weight':
                 #    ones=indicator.sum()
                 #    length=indicator.shape[0]*indicator.shape[1]*indicator.shape[2]*indicator.shape[3]
                 #    print(ones/length)
                 indicator=torch.where(indicator!=0,indicator,torch.tensor(-1,dtype=torch.float32).to(args.device))
                 #singleUserIndicator[k]*=0
                 #singleUserIndicator[k]+=indicator
                 #distributions2=torch.full(distributions.shape,0.5).to(args.device)
                 #m2=binomial.Binomial(1,distributions2)
                 #indicator2=m2.sample()
                 #indicator+=indicator2*indicator
                 singleUserIndicator[k]*=0.0
                 singleUserIndicator[k]+=indicator
                 #net.state_dict()[k]*=0
                 #net.state_dict()[k]+=indicator*R*((exp(epslon)+1)/(exp(epslon)-1))+C
                 #net.state_dict()[k]+=indicator
                 #net.state_dict()[k]=(net.state_dict()[k]-(1/4)*C)*(4/3)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),singleUserIndicator

