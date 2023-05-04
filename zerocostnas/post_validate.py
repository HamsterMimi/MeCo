import sys
import os 
import json
import tqdm
import torch
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
import glob
import logging
import shutil
import numpy as np
sys.path.insert(0, '../')
from nasbench201.cell_infers.tiny_network import TinyNetwork
from nasbench201.genotypes import Structure
from nas_201_api import NASBench201API as API
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.nas.genotypes import Genotype
import nasbench201.utils as ig_utils
from foresight.pruners import *
from Scorers.scorer import Jocab_Scorer
import torchvision.transforms as transforms
import argparse
from mobilenet_search_space.retrain_architecture.model import Network
from torch.utils.tensorboard import SummaryWriter
from sota.cnn.hdf5 import H5Dataset
parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')                    
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--save_path', type=str, default='../experiments/sota', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--ckpt_path', type=str, help='path that saved networks pool')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--maxiter', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for alpha')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--validate_rounds', type=int, default=10, help='score round for networks')
parser.add_argument('--proj_crit', type=str, default='jacob', choices=['loss', 'acc', 'var', 'cor', 'norm', 'jacob', 'snip', 'fisher', 'synflow', 'grad_norm', 'grasp', 'jacob_cov', 'comb', 'meco', 'zico'], help='criteria for projection')
parser.add_argument('--edge_decision', type=str, default='random', choices=['random','reverse', 'order', 'global_op_greedy', 'global_op_once', 'global_edge_greedy', 'global_edge_once'], help='which edge to be projected next')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def load_network_pool(ckpt_path):
    with open(os.path.join(ckpt_path,'networks_pool.json'), 'r') as save_file:
        for line in save_file:
            networks_pool = json.loads(line)
        if 'pool_size' in networks_pool:
            return networks_pool['search_space'], networks_pool['dataset'],  networks_pool['networks'], networks_pool['pool_size']
        else:
            return networks_pool['search_space'], networks_pool['dataset'],  networks_pool['networks'], len(networks_pool['networks'])
#### args augment
search_space, dataset, networks_pool, pool_size = load_network_pool(args.ckpt_path)
# print(search_space, dataset, networks_pool, pool_size)
search_space = search_space.strip()
dataset = dataset.strip()
expid = args.save

args.save = '{}/{}-valid-{}-{}-{}-{}'.format(args.save_path, search_space, args.save, args.seed, pool_size, args.validate_rounds)
if not dataset == 'cifar10':
    args.save += '-' + dataset
if not args.edge_decision == 'random':
    args.save += '-' + args.edge_decision
if not args.proj_crit == 'jacob':
    args.save += '-' + args.proj_crit
scripts_to_save = glob.glob('*.py') + ['../exp_scripts/{}.sh'.format(expid)]
if os.path.exists(args.save):
    if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        print('proceed to override saving directory')
        shutil.rmtree(args.save)
    else:
        exit(0)
ig_utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

log_file = 'log'
log_file += '.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)
logging.info('load pool from space:%s and dataset:%s', search_space, dataset)

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
if dataset == 'cifar100':
    n_classes = 100
elif dataset == 'imagenet16-120':
    n_classes = 120
elif dataset == 'imagenet':
    n_classes = 1000
else:
    n_classes = 10
    
if search_space == 'nas-bench-201':
    api = API('../data/NAS-Bench-201-v1_0-e61699.pth')

if search_space == 'nb_macro':
    import pickle as pkl
    f = open('../data/nbmacro-base-0.pickle','rb')
    head = pkl.load(f)
    value = pkl.load(f)
    api ={}
    for v in value:
        h, val_t1, test_t1, t_time = v
        api[h] = test_t1
def main():
    #### data
    if dataset == 'imagenet':
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

        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])
        train_data = H5Dataset(os.path.join(args.data, 'imagenet-train-256.h5'), transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.validate_rounds * args.batch_size))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))

    else:
        if dataset == 'cifar10':
            train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        elif dataset == 'cifar100':
            train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        elif dataset == 'svhn':
            train_transform, valid_transform = ig_utils._data_transforms_svhn(args)
            train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
            valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
        elif dataset == 'imagenet16-120':
            from nasbench201.DownsampledImageNet import ImageNet16
            mean = [x / 255 for x in [122.68, 116.66, 104.01]]
            std = [x / 255 for x in [63.22,  61.26, 65.09]]
            lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
            train_transform = transforms.Compose(lists)
            train_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
            valid_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
            assert len(train_data) == 151700

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.validate_rounds * args.batch_size))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=4)

    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)

    if args.proj_crit == 'jacob':
        validate_scorer = Jocab_Scorer(gpu)

    best_id = None
    best_score = 0
    best_networks = None
    crit_list = []
    print(len(train_queue))
    net_history = []
    for net_config in tqdm.tqdm(networks_pool, desc="networks", position=0):
        net_id = net_config['id']
        # print(net_id)
        net_genotype = net_config['genotype']
        # print(net_genotype)
        if net_genotype not in net_history:
            net_history.append(net_genotype)
            # print(net_genotype)
            network = get_networks_from_genotype(net_genotype, dataset, search_space)
            # print(network)
            if args.proj_crit == 'jacob':
                validate_scorer.setup_hooks(network, args.batch_size)
            for step, (input, target) in tqdm.tqdm(enumerate(train_queue), desc="validate_rounds", position=1, leave=False):
                input.cuda()
                target.cuda()
                if args.proj_crit == 'jacob':
                    score = validate_scorer.score(network, input, target)
                else:
                    #score =  score_loop(network, None, train_queue, args.gpu, None, args.proj_crit)
                    network.requires_feature = False
                    measures = predictive.find_measures(network,
                                                        train_queue,
                                                        ('random', 1, n_classes),
                                                        torch.device("cuda"),
                                                        measure_names=[args.proj_crit])



                    # measures = predictive.find_measures(network,
                    #                 train_queue,
                    #                 ('random', 1, n_classes), #TODO don't hard-code num_classes to 10
                    #                 torch.device("cuda"),
                    #                 measure_names=[args.proj_crit])
                    score = measures[args.proj_crit]

                if step == 0:
                    crit_list.append(score)
                else:
                    crit_list[-1] += score
                if args.proj_crit != 'jacob':
                    break
    #best_networks = networks_pool[np.nanargmax(crit_list)]['genotype']
    best_networks = net_history[np.nanargmax(crit_list)]

    if search_space == 'nas-bench-201':
        cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = query(api, best_networks, logging)

    networks_info={}
    networks_info['search_space'] = search_space
    networks_info['dataset'] = dataset
    networks_info['networks'] = best_networks
    with open(os.path.join(args.save,'best_networks.json'), 'w') as save_file:
        json.dump(networks_info, save_file)


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

def get_networks_from_genotype(genotype_str, dataset, search_space):
    if search_space == 'nas-bench-201':
        net_index = api.query_index_by_arch(genotype_str)
        ##print(dataset)
        net_config = api.get_net_config(net_index, 'cifar10-valid')
        print(net_config)
        genotype = Structure.str2structure(net_config['arch_str'])
        network = TinyNetwork(net_config['C'], net_config['N'], genotype, n_classes)
        return network
    elif search_space == 'mobilenet':
        rngs = [int(id) for id in genotype_str.split(' ')]
        network = Network(rngs, n_class=n_classes)
        return network
    else:
        # print(genotype_str)
        genotype_config = json.loads(genotype_str)
        genotype = Genotype(normal=genotype_config['normal'], normal_concat=genotype_config['normal_concat'], reduce=genotype_config['reduce'], reduce_concat=genotype_config['reduce_concat'])

        if dataset == 'imagenet':
            network = NetworkImageNet(args.init_channels, n_classes, args.layers, False, genotype)
        else:
            network = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype)
        network.drop_path_prob = 0.
        return network


if __name__ == '__main__':
    main()
