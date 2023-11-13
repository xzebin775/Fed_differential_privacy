import copy

import torch
from utils.options import args_parser
from math import exp


def perturb_initial(net_para, indicator, w_C, w_R, k):
    args = args_parser()

    epslon = args.epslon
    if args.method == 'modified':
        net_para[k] *= 0
        net_para[k] += w_R[k] * ((exp(epslon) + 1) / (2 * exp(epslon) - 2)) * indicator * (4 / 3)
        net_para[k] += w_C[k]

    elif args.method == 'duchi':
        net_para[k] *= 0
        net_para[k] += w_R[k] * ((exp(epslon) + 1) / (1 * exp(epslon) - 1)) * indicator
        net_para[k] += w_C[k]

    elif args.method == 'modified2':
        net_para[k] *= 0
        indicator_tmp = torch.where(torch.abs(indicator) == 2.0, indicator / 2, indicator)
        net_para[k] += (w_R[k] / 2) * ((exp(epslon) + 1) / (1 * exp(epslon) - 1)) * indicator_tmp
        w_C1 = copy.deepcopy(w_C[k])
        w_C1 -= w_R[k] / 2
        w_C2 = copy.deepcopy(w_C[k])
        w_C2 += w_R[k] / 2
        net_para[k] += torch.where(torch.abs(indicator) == 2.0, w_C2, w_C1)
        # if k=='conv1.weight':
        #    print("conv1.weight[0][0][0][0]为：{}".format(net_para[k][0][0][0][0]))


def perturb_reduce(net_para, w_C, w_R, k):
    args = args_parser()
    net_tmp = copy.deepcopy(net_para)

    epslon = args.epslon
    if args.method == 'modified':
        net_para[k] *= 0
        net_para[k] += w_R[k] * ((exp(epslon) + 1) / (2 * exp(epslon) - 2)) * net_tmp[k]
        net_para[k] += w_C[k] * (3 / 4)
        net_para[k] *= (4 / 3)

    elif args.method == 'duchi':
        net_para[k] *= 0
        net_para[k] += w_R[k] * ((exp(epslon) + 1) / (1 * exp(epslon) - 1)) * net_tmp[k]
        net_para[k] += w_C[k]
