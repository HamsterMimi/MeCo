import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import random
import numpy as np
import torch
import shutil
import nasbench201.utils as ig_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import json
import copy

from sota.cnn.model_search import Network as DartsNetwork
from sota.cnn.model_search_darts_proj import DartsNetworkProj
from sota.cnn.model_search_imagenet_proj import ImageNetNetworkProj
# from optimizers.darts.architect import Architect as DartsArchitect
from nasbench201.architect_ig import Architect
from sota.cnn.spaces import spaces_dict
from foresight.pruners import *

from torch.utils.tensorboard import SummaryWriter
from sota.cnn.init_projection import pt_project
from hdf5 import H5Dataset

torch.set_printoptions(precision=4, sci_mode=False)

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--seed', type=int, default=666, help='random seed')

#model opt related config 
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--nesterov', action='store_true', default=True, help='using nestrov momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

#system config
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--save_path', type=str, default='../../experiments/sota', help='experiment name')
#search sapce config
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
parser.add_argument('--pool_size', type=int, default=10, help='number of model to proposed')

## projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random','reverse', 'order', 'global_op_greedy', 'global_op_once', 'global_edge_greedy', 'global_edge_once', 'sample'], help='used for both proj_op and proj_edge')
parser.add_argument('--proj_crit_normal', type=str, default='meco', choices=['loss', 'acc', 'jacob', 'comb', 'synflow', 'snip', 'fisher', 'var', 'cor', 'norm', 'grad_norm', 'grasp', 'jacob_cov', 'meco', 'zico'])
parser.add_argument('--proj_crit_reduce', type=str, default='meco', choices=['loss', 'acc', 'jacob', 'comb', 'synflow', 'snip', 'fisher', 'var', 'cor', 'norm', 'grad_norm', 'grasp', 'jacob_cov', 'meco', 'zico'])
parser.add_argument('--proj_crit_edge',   type=str, default='meco', choices=['loss', 'acc', 'jacob', 'comb', 'synflow', 'snip', 'fisher', 'var', 'cor', 'norm', 'grad_norm', 'grasp', 'jacob_cov', 'meco', 'zico'])
parser.add_argument('--proj_mode_edge', type=str, default='reg', choices=['reg'],
                    help='edge projection evaluation mode, reg: one edge at a time')
args = parser.parse_args()

#### args augment

expid = args.save
args.save = '{}/{}-search-{}-{}-{}-{}-{}'.format(args.save_path,
    args.dataset, args.save, args.search_space, args.seed, args.pool_size, args.proj_crit_normal)

if not args.edge_decision == 'random':
    args.save += '-' + args.edge_decision

scripts_to_save = glob.glob('*.py') + glob.glob('../../nasbench201/architect*.py') + glob.glob('../../optimizers/darts/architect.py')
if os.path.exists(args.save):
    if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        print('proceed to override saving directory')
        shutil.rmtree(args.save)
    else:
        exit(0)
ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

#### logging
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

if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet':
    n_classes = 1000
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
    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)

    #### model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    ## darts
    if args.dataset == 'imagenet':
        model = ImageNetNetworkProj(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    else:
        model = DartsNetworkProj(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    model = model.cuda()
    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))
    
    #### data
    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
        ])
        #for test
        #from nasbench201.DownsampledImageNet import ImageNet16
        # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        # n_classes = 10
        train_data = H5Dataset(os.path.join(args.data, 'imagenet-train-256.h5'), transform=train_transform)
        #valid_data  = H5Dataset(os.path.join(args.data, 'imagenet-val-256.h5'),   transform=test_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    else:
        if args.dataset == 'cifar10':
            train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        elif args.dataset == 'cifar100':
            train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        elif args.dataset == 'svhn':
            train_transform, valid_transform = ig_utils._data_transforms_svhn(args)
            train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
            valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)
    # for x, y in train_queue:
    #     from torchvision import transforms
    #     unloader = transforms.ToPILImage()
    #     image = x.cpu().clone()  # clone the tensor
    #     image = image.squeeze(0)  # remove the fake batch dimension
    #     image = unloader(image)
    #     image.save('example.jpg')

        # print(x.size())
        # exit()


    #### projection
    networks_pool={}
    networks_pool['search_space'] = args.search_space
    networks_pool['dataset'] = args.dataset
    networks_pool['networks'] = []
    for i in range(args.pool_size):
        network_info={}
        logging.info('{} MODEL HAS SEARCHED'.format(i+1))
        pt_project(train_queue, model, args)

        ## logging
        num_params = ig_utils.count_parameters_in_Compact(model)
        genotype = model.genotype()
        json_data = {}
        json_data['normal'] = genotype.normal
        json_data['normal_concat'] = [x for x in genotype.normal_concat]
        json_data['reduce'] = genotype.reduce
        json_data['reduce_concat'] = [x for x in genotype.reduce_concat] 
        json_string = json.dumps(json_data)
        logging.info(json_string)
        network_info['id'] = str(i)
        network_info['genotype'] = json_string
        networks_pool['networks'].append(network_info)
        model.reset_arch_parameters()

    with open(os.path.join(args.save,'networks_pool.json'), 'w') as save_file:
        json.dump(networks_pool, save_file)

if __name__ == '__main__':
    main()