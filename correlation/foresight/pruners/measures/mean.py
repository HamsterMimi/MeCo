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
        s = time.time()
        mean = torch.mean(data_input[0])
        e = time.time()
        t = e - s
        result_list.append(mean)
        result_list.append(t)
    net.classifier.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        # t1 = time.time()
        y = net(x[st:en])
        # t2 = time.time()
        # print('var:', t2-t1)
    m = result_list[0].item()
    t = result_list[1]
    result_list.clear()
    return m, t



@measure('mean', bn=True)
def compute_mean(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    # print('var:', features.shape)
    try:
        mean, t = get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        print(e)
        mean, t = np.nan, None
    # print(jc)
    # print(f'var time: {t} s')
    return mean, t
