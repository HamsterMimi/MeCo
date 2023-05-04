import os
import sys
sys.path.insert(0, '../')
import time
import glob
import json
import shutil
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import nasbench201.utils as ig_utils
from nasbench201.search_model_darts_proj import TinyNetworkDartsProj
from nasbench201.cell_operations import SearchSpaceNames
from nasbench201.init_projection import pt_project, global_op_greedy_pt_project, global_op_once_pt_project, global_edge_greedy_pt_project, global_edge_once_pt_project, shrink_pt_project, tenas_project
from nas_201_api import NASBench201API as API

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


parser = argparse.ArgumentParser("sota")
# data related 
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120'], help='choose dataset')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')

#search space setting
parser.add_argument('--search_space', type=str, default='nas-bench-201')

parser.add_argument('--pool_size', type=int, default=100, help='number of model to proposed')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')

#system configurations
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--save', type=str, default='exp', help='experiment name')

#default opt setting for model
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--nesterov', action='store_true', default=True, help='using nestrov momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

#### common
parser.add_argument('--fast', action='store_true', default=True, help='skip loading api which is slow')

#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random','reverse', 'order', 'global_op_greedy', 'global_op_once', 'global_edge_greedy', 'global_edge_once', 'shrink_pt_project'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default="comb", choices=['loss', 'acc', 'jacob', 'snip', 'fisher', 'synflow', 'grad_norm', 'grasp', 'jacob_cov','tenas', 'var', 'cor', 'norm', 'comb', 'meco'], help='criteria for projection')
args = parser.parse_args()

#### args augment
expid = args.save
args.save = '../experiments/nas-bench-201/prop-{}-{}-{}'.format(args.save, args.seed, args.pool_size)
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if not args.edge_decision == 'random':
    args.save += '-' + args.edge_decision
if not args.proj_crit == 'jacob':
    args.save += '-' + args.proj_crit

#### logging
scripts_to_save = glob.glob('*.py') \
                  # + ['../exp_scripts/{}.sh'.format(expid)]
if os.path.exists(args.save):
    if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        print('proceed to override saving directory')
        shutil.rmtree(args.save)
    else:
        exit(0)
ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

log_file = 'log.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)

if os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else:
        exit(0)

fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

#### macros
if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10

def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info('gpu device = %d' % gpu)

    #### model
    criterion = nn.CrossEntropyLoss()
    search_space = SearchSpaceNames[args.search_space]

    # 初始化超网络
    model = TinyNetworkDartsProj(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    model_thin = TinyNetworkDartsProj(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args, stem_channels=1)
    model = model.cuda()
    model_thin = model_thin.cuda()
    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))

    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from nasbench201.DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)


    #format network pool diction
    networks_pool={}
    networks_pool['search_space'] = args.search_space
    networks_pool['dataset'] = args.dataset
    networks_pool['networks'] = []
    networks_pool['pool_size'] = args.pool_size 
    #### architecture selection / projection
    for i in range(args.pool_size):
        network_info={}
        logging.info('{} MODEL HAS SEARCHED'.format(i+1))
        if args.edge_decision == 'global_op_greedy':
            global_op_greedy_pt_project(train_queue, model, args)
        elif args.edge_decision == 'global_op_once': 
            global_op_once_pt_project(train_queue, model, args)
        elif args.edge_decision == 'global_edge_greedy':
            global_edge_greedy_pt_project(train_queue, model, args)
        elif args.edge_decision == 'global_edge_once':
            global_edge_once_pt_project(train_queue, model, args)
        elif args.edge_decision == 'shrink_pt_project':
            shrink_pt_project(train_queue, model, args)
            api = API('../data/NAS-Bench-201-v1_0-e61699.pth')
            cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = query(api, model.genotype().tostr(), logging)
        else:
            if args.proj_crit == 'jacob':
                pt_project(train_queue, model, args)
            else:
                pt_project(train_queue, model, args)
                # tenas_project(train_queue, model, model_thin, args)

        network_info['id'] = str(i)
        network_info['genotype'] = model.genotype().tostr()
        networks_pool['networks'].append(network_info)
        model.reset_arch_parameters()
    
    with open(os.path.join(args.save,'networks_pool.json'), 'w') as save_file:
        json.dump(networks_pool, save_file)


#### util functions
def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def query(api, genotype, logging):
    result = api.query_by_arch(genotype, hp='200')
    logging.info('{:}'.format(result))
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
    logging.info('cifar10 train %f test %f', cifar10_train, cifar10_test)
    logging.info('cifar100 train %f valid %f test %f', cifar100_train, cifar100_valid, cifar100_test)
    logging.info('imagenet16 train %f valid %f test %f', imagenet16_train, imagenet16_valid, imagenet16_test)
    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    main()
