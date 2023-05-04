import gc
import numpy as np
import os
import sys
import torch
import torch.nn.functional as f
from operator import mul
from functools import reduce
import copy
sys.path.insert(0, '../')

def Jocab_Score(ori_model, input, target, weights=None):
    model = copy.deepcopy(ori_model)
    model.eval()
    model.proj_weights = weights
    num_edge, num_op = model.num_edge, model.num_op
    for i in range(num_edge):
        model.candidate_flags[i] = False
    batch_size = input.shape[0]
    model.K = torch.zeros(batch_size, batch_size).cuda()

    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            model.K = model.K + K + K2
        except:
            pass

    for name, module in model.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
    
    input = input.cuda()
    model(input)
    score = hooklogdet(model.K.cpu().numpy())
    del model
    del input
    return score

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

# NTK
#------------------------------------------------------------
#https://github.com/VITA-Group/TENAS/blob/main/lib/procedures/ntk.py
#
def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network

def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1, weights=None):
    device = torch.cuda.current_device()
    ntks = []
    copied_networks = []
    for network in networks:
        network = network.cuda(device=device)
        net = copy.deepcopy(network)
        net.proj_weights = weights
        num_edge, num_op = net.num_edge, net.num_op
        for i in range(num_edge):
            net.candidate_flags[i] = False
        if train_mode:
            net.train()
        else:
            net.eval()
        copied_networks.append(net)
    ######
    grads = [[] for _ in range(len(copied_networks))]
    for i, (inputs, targets) in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(copied_networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    
    del copied_networks
    return conds