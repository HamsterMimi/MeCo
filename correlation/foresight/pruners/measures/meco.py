# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import copy
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
from torch import nn

from . import measure


def get_score(net, x, target, device, split_data):
    result_list = []
    x = torch.randn(size=(1, 3, 64, 64)).to(device)
    net.to(device)
    def forward_hook(module, data_input, data_output):

        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        n = fea.shape[0]
        corr = torch.corrcoef(fea)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = torch.min(torch.real(values))
        result_list.append(result)

    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
        # break
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()



@measure('meco', bn=True)
def compute_meco(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()


    try:
        meco = get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        print(e)
        meco = np.nan, None
    return meco
