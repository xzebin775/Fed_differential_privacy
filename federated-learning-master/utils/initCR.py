import torch
import copy
import re
import math
from utils.options import args_parser
import torch.nn as nn

def init_cr(net,is_init):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if is_init:
        w_C = copy.deepcopy(net.state_dict())
        for k in w_C.keys():
            w_C[k] *= 0
            print(w_C[k].device)
        w_R = copy.deepcopy(net.state_dict())
        ###############################
        fan_in = 0.0
        i = 1
        for k in w_R.keys():
            if (not (re.match(r"conv", k) == None)):
                if (i % 2 != 0):
                    fan_in = (w_R[k].shape)[1] * (w_R[k].shape)[2] * (w_R[k].shape)[3]
                w_R[k] *= 0
                w_R[k] += torch.full(w_R[k].shape, 1 / math.sqrt(fan_in)).to(args.device)

            else:
                if (i % 2 != 0):
                    fan_in = (w_R[k].shape)[1]
                w_R[k] *= 0
                w_R[k] += torch.full(w_R[k].shape, 1 / math.sqrt(fan_in)).to(args.device)
            i += 1
    else:
        return copy.deepcopy(net).state_dict(),copy.deepcopy(net).state_dict()



