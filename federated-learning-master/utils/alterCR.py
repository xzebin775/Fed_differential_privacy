import torch
import copy
from utils.options import args_parser


def alterCR(avg_indicator,w_C,w_avg):
    args = args_parser()
    alpha = args.alpha
    cr_indicator = copy.deepcopy(avg_indicator)
    for k in avg_indicator.keys():
        cr_indicator[k] *= 0
        tmp_indicator = torch.where(avg_indicator[k] > alpha, 1, avg_indicator[k])
        cr_indicator[k] = torch.where(tmp_indicator < -alpha, 1, tmp_indicator)

        tmp_w=torch.where(cr_indicator==1,w_C[k])
