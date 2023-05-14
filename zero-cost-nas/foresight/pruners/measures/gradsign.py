# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
from torch import nn
import numpy as np

from . import measure


def get_flattened_metric(net, metric):
    grad_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grad_list.append(metric(layer).flatten())
    flattened_grad = np.concatenate(grad_list)

    return flattened_grad


def get_grad_conflict(net, inputs, targets, loss_fn):
    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad()
        outputs = net.forward(inputs[[i]])
        loss = loss_fn(outputs, targets[[i]])
        loss.backward()
        flattened_grad = get_flattened_metric(net, lambda
            l: l.weight.grad.data.clone().cpu().numpy() if l.weight.grad is not None else torch.zeros_like(
            l.weight).clone().cpu().numpy())
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nansum(direction_code)
    return score


def get_gradsign(input, target, net, device, loss_fn):
    s = []
    net = net.to(device)
    x, target = input, target
    # x2 = torch.clone(x)
    # x2 = x2.to(device)
    x, target = x.to(device), target.to(device)
    s.append(get_grad_conflict(net=net, inputs=x, targets=target, loss_fn=loss_fn))
    s = np.mean(s)
    return s

@measure('gradsign', bn=True)
def compute_gradsign(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()


    try:
        gradsign = get_gradsign(inputs, targets, net, device, loss_fn)
    except Exception as e:
        print(e)
        gradsign= np.nan

    return gradsign
