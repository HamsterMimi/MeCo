import time
import torch

from . import measure
from ..p_utils import get_layer_metric_array



@measure('param_count', copy_net=False, mode='param')
def get_param_count_array(net, inputs, targets, mode, loss_fn, split_data=1):
    s = time.time()
    count = get_layer_metric_array(net, lambda l: torch.tensor(sum(p.numel() for p in l.parameters() if p.requires_grad)), mode=mode)
    e = time.time()
    t = e - s
    # print(f'param_count time: {t} s')
    return count, t