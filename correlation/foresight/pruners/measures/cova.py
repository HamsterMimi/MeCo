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
from torch import nn

from . import measure


def get_score(net, x, target, device, split_data):
    result_list = []
    result_t = []
    def forward_hook(module, data_input, data_output):
        s = time.time()
        fea = data_output[0].detach().cpu().numpy()
        fea = fea.reshape(fea.shape[0], -1)
        result = 1 / np.var(np.corrcoef(fea))
        e = time.time()
        t = e - s
        result_list.append(result)
        result_t.append(t)
    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)



    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    results = np.array(result_list)
    results = results[np.logical_not(np.isnan(results))]
    v = np.sum(results)
    t = sum(result_t)
    result_list.clear()
    result_t.clear()
    return v, t



@measure('cova', bn=True)
def compute_cova(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    try:
        cova, t = get_score(net, inputs, targets, device, split_data=split_data)
    except Exception as e:
        print(e)
        cova, t = np.nan, None
    return cova, t
