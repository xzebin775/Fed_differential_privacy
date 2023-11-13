import torch.distributions.binomial as binomial
import torch
import copy
from math import exp
from utils.options import args_parser
from torch import maximum, minimum


def get_distributions(net_para, w_C, w_R, k):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    epslon = args.epslon
    # 对distributions做整理
    distributions = copy.deepcopy(net_para[k])
    distributions = torch.div((distributions - w_C[k]), w_R[k]) * ((exp(epslon) - 1) / (2 * exp(epslon) + 2)) + 0.5

    distributions = torch.where(torch.isinf(distributions), torch.full_like(distributions, 1.0), distributions)
    distributions = torch.where(torch.isnan(distributions), torch.full_like(distributions, 1.0), distributions)
    distributions = minimum(distributions, torch.full_like(distributions, 1.0))
    distributions = maximum(distributions, torch.full_like(distributions, 0))

    if args.method == 'modified':
        # 设置第二个distributions分布
        distributions2 = torch.full(distributions.shape, 0.5).to(args.device)
        m2 = binomial.Binomial(1, distributions2)
        indicator2 = m2.sample()
        m = binomial.Binomial(1, distributions)
        indicator = m.sample()
        indicator = torch.where(indicator != 0, indicator, torch.tensor(-1, dtype=torch.float32).to(args.device))
        indicator += indicator2 * indicator
    elif args.method == 'duchi':
        m = binomial.Binomial(1, distributions)
        indicator = m.sample()
        indicator = torch.where(indicator != 0, indicator, torch.tensor(-1, dtype=torch.float32).to(args.device))
    return indicator


def get_distributions2(net_para, w_C, w_R, k):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    epslon = args.epslon

    indicator1 = torch.where(net_para[k] < w_C[k], torch.tensor(0.0, dtype=torch.float32).to(args.device),
                             torch.tensor(1.0, dtype=torch.float32).to(args.device))
    distributions = copy.deepcopy(net_para[k])
    w_C1 = copy.deepcopy(w_C[k])
    w_C1 -= w_R[k] / 2
    w_C2 = copy.deepcopy(w_C[k])
    w_C2 += w_R[k] / 2
    dist_tmp1 = torch.div((distributions - w_C1), w_R[k] / 2) * ((exp(epslon) - 1) / (2 * exp(epslon) + 2)) + 0.5
    dist_tmp2 = torch.div((distributions - w_C2), w_R[k] / 2) * ((exp(epslon) - 1) / (2 * exp(epslon) + 2)) + 0.5
    distributions = torch.where(indicator1 == 0.0, dist_tmp1, dist_tmp2)

    distributions = torch.where(torch.isinf(distributions), torch.full_like(distributions, 1.0), distributions)
    distributions = torch.where(torch.isnan(distributions), torch.full_like(distributions, 1.0), distributions)
    distributions = minimum(distributions, torch.full_like(distributions, 1.0))
    distributions = maximum(distributions, torch.full_like(distributions, 0))

    m = binomial.Binomial(1, distributions)
    indicator2 = m.sample()
    indicator2 = torch.where(indicator2 != 0, indicator2, torch.tensor(-1, dtype=torch.float32).to(args.device))
    indicator2 += indicator1 * indicator2
    return indicator2
