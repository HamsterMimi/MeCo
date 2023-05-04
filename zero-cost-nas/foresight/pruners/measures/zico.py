# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import time

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch

from . import measure
from torch import nn

from ...dataset import get_cifar_dataloaders


def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                try:
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
                except:
                    continue
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                try:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
                except:
                    continue
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        # print(grad_dict[modname].shape)
        # print(grad_dict[modname].shape, nsr_std.shape)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


def getzico(network, inputs, targets, loss_fn, split_data=2):
    grad_dict = {}
    network.train()
    device = inputs.device
    network.to(device)
    N = inputs.shape[0]
    split_data = 2

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        outputs = network.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
        grad_dict = getgrad(network, grad_dict, sp)
    # print(grad_dict)
    res = caculate_zico(grad_dict)
    return res





@measure('zico', bn=True)
def compute_zico(net, inputs, targets, split_data=2, loss_fn=None):

    # Compute gradients (but don't apply them)
    net.zero_grad()

    # print('var:', feature.shape)
    try:
        zico = getzico(net, inputs, targets, loss_fn, split_data=split_data)
    except Exception as e:
        print(e)
        zico= np.nan
    # print(jc)
    # print(f'var time: {t} s')
    return zico