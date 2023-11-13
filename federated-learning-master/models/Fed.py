#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn, maximum, minimum
from math import exp


def FedAvg(w,w_indicator,need):
#def FedAvg(w,need):
    epslon=3.0
    ##对权重进行聚合的同时，计算出本轮次的C和R
    w_avg = copy.deepcopy(w[0])
    w_max=copy.deepcopy(w[0])
    w_min=copy.deepcopy(w[0])
    #w_RTest=copy.deepcopy(w[0])
    w_avgIndicator=copy.deepcopy(w_indicator[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            w_avgIndicator[k]+=w_indicator[i][k]
            w_max[k]=maximum(w_max[k],w[i][k])
            w_min[k]=minimum(w_min[k],w[i][k])
        w_avg[k] = torch.div(w_avg[k], len(w))
        #w_RTest[k]=torch.div(w_RTest[k],len(w))
        #w_avgIndicator[k]=torch.div(w_avgIndicator[k],len(w))
        #print(w_avg[k])
        #w_avgIndicator[k]*=w_R[k]*((exp(epslon)+1)/(1*exp(epslon)-1))
        #w_avgIndicator[k]+=w_C[k]
        if k=='conv1.weight':
            print("w_avg is {}".format(w_avg[k][0][0][0][0]))
            #print("w_avgIndicator is {}".format(w_avgIndicator[k][0][0][0][0]))
        #w_avg[k]*=0.0
        #w_avg[k]+=w_avgIndicator[k]*w_R[k]*((exp(epslon)+1)/(exp(epslon)-1))+w_C[k]
        #w_RTest[k]*=0.0
        #w_RTest[k]+=torch.where(((w_avgIndicator[k]<0.8)&(w_avgIndicator[k]>-0.8)),w_R[k],w_R[k]*(3.0/2.0))
        #w_avg[k]*=w_R[k]
        #w_avg[k]*=((exp(epslon)+1)/(exp(epslon)-1))
        #w_avg[k]+=w_C[k]
        #print(w_avg[k])
    if need:
        w_C=copy.deepcopy(w[0])
        w_R=copy.deepcopy(w[0])
        for k in w_C.keys():
            w_C[k]*=0
            w_C[k]+=(w_max[k]+w_min[k])/2

            w_R[k]*=0
            w_R[k]+=(w_max[k]-w_C[k])
        return w_avg,w_C,w_R
    return w_avg,w_C,w_R
