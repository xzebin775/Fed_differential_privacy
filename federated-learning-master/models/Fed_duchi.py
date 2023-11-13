#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import maximum, minimum
from math import exp
from utils.options import args_parser
from utils.perturbation import perturb_reduce
from utils.alterCR import alterCR


def FedAvg(w, w_C, w_R, iter):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    epslon = args.epslon
    #对权重进行聚合的同时，计算出本轮次的C和R
    w_avg = copy.deepcopy(w[0])

    # iter=0时对w_C和w_R进行初始化
    if iter==0:
        w_max = copy.deepcopy(w[0])
        w_min = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
                w_max[k] = maximum(w_max[k], w[i][k])
                w_min[k] = minimum(w_min[k], w[i][k])
            w_avg[k] = torch.div(w_avg[k], len(w))
            if k == 'conv1.weight':
                print("w_avg is {}".format(w_avg[k][0][0][0][0]))

        for k in w_C.keys():
            if w_C[k].shape == torch.Size([]):
                continue
            w_C[k] *= torch.tensor([0.0], dtype=torch.float32).to(args.device)
            w_C[k] += (w_max[k] + w_min[k]) / 2

            w_R[k] *= torch.tensor([0.0], dtype=torch.float32).to(args.device)
            w_R[k] += (w_max[k] - w_C[k])
        return w_avg, w_C, w_R

    # 当iter不为0是，对w_R每隔五轮进行一次扩大
    else:
        if args.communication == 'initial':
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))

                if k == 'conv1.weight':
                    print("w_avg is {}".format(w_avg[k][0][0][0][0]))


        elif args.communication == 'reduce':
            for k in w_avg.keys():
                if w_avg[k].shape == torch.Size([]):
                    continue
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
                # avg_indicator=copy.deepcopy(w_avg)
                perturb_reduce(w_avg, w_C, w_R, k)
                # w_C[k]*=torch.tensor([0.0],dtype=torch.float32).to(args.device)
                # print(w_C[k][0][0][0][0])
                # print(w_R[k][0][0][0][0])
                # print(w_avg[k][0][0][0][0])
                # w_C[k]+=w_avg[k]+w_R[k]*(1/10)
                if k == 'conv1.weight':
                    print("w_avg is {}".format(w_avg[k][0][0][0][0]))
        # if iter % 15 == 0:
        #     for k in w_R.keys():
        #         if w_R[k].shape == torch.Size([]):
        #             continue
        #         w_R[k] *= ((exp(epslon) + 1) / (exp(epslon) - 1))
        return w_avg, w_avg, w_R
