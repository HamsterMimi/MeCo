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


def get_score(net, x, target, device, split_data):
    result_list = []
    def forward_hook(module, data_input, data_output):
        var = torch.var(data_input[0])
        result_list.append(var)
    net.classifier.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    v = result_list[0].item()
    result_list.clear()
    return v



@measure('var', bn=True)
def compute_var(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    # print('var:', feature.shape)
    try:
        var= get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        print(e)
        var= np.nan
    # print(jc)
    # print(f'var time: {t} s')
    return var