import argparse
import os

import time

from foresight.dataset import *
from foresight.models import nasbench2
from foresight.pruners import predictive
from foresight.weight_initializers import init_net
from models import get_cell_based_tiny_net
import pickle


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='../data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero, one]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='ImageNet16-120',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=5, help='GPU index to work on')
    parser.add_argument('--data_size', type=int, default=32, help='data_size')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='appoint', help='random, grasp, appoint supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args.device)

    if args.noacc:
        api = pickle.load(open(args.api_loc,'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, resize=args.data_size)
    x, y = next(iter(train_loader))
    # random data
    # x = torch.rand((args.batch_size, 3, args.data_size, args.data_size))
    # y = 0

    cached_res = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'nb2_{args.search_space}_{pre}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}_{args.batch_size}.p'
    op = os.path.join(args.outdir, pfn)

    end = len(api) if args.end == 0 else args.end

    # loop over nasbench2 archs
    for i, arch_str in enumerate(api):

        if i < args.start:
            continue
        if i >= end:
            break

        res = {'i': i, 'arch': arch_str}
        # print(arch_str)
        if args.search_space == 'tss':
            net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
            arch_str2 = nasbench2.get_arch_str_from_model(net)
            if arch_str != arch_str2:
                print(arch_str)
                print(arch_str2)
                raise ValueError
        elif args.search_space == 'sss':
            config = api.get_net_config(i, args.dataset)
            # print(config)
            net = get_cell_based_tiny_net(config)
        net.to(args.device)
        # print(net)

        init_net(net, args.init_w_type, args.init_b_type)

        # print(x.size(), y)
        measures = get_score(net, x, i, args.device)

        res['meco'] = measures

        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                     hp='200', is_random=False)

            trainacc = info['train-accuracy']
            valacc = info['valid-accuracy']
            testacc = info['test-accuracy']

            res['trainacc'] = trainacc
            res['valacc'] = valacc
            res['testacc'] = testacc

        print(res)
        cached_res.append(res)

        # write to file
        if i % args.write_freq == 0 or i == len(api) - 1 or i == 10:
            print(f'writing {len(cached_res)} results to {op}')
            pf = open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []
