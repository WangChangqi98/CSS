import argparse
import random
from typing import Iterable, Union
from copy import deepcopy as dcopy
from typing import List, Set
import collections
from functools import partial, reduce
import torch
import numpy as np
import os
import datetime
# from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn
import sys

##### Hyper Parameters Define #####

def _parser_(input_strings: str) -> Union[dict, None]:
    if input_strings.__len__() == 0:
        return None
    assert input_strings.find('=') > 0, f"Input args should include '=' to include value"
    keys, value = input_strings.split('=')[:-1][0].replace(' ', ''), input_strings.split('=')[1].replace(' ', '')
    keys = keys.split('.')
    keys.reverse()
    for k in keys:
        d = {}
        d[k] =value
        value = dcopy(d)
    return dict(value)

def _parser(strings: List[str]) -> List[dict]:
    assert isinstance(strings, list)
    args: List[dict] = [_parser_(s) for s in strings]
    args = reduce(lambda x, y: dict_merge(x, y, True), args)
    return args

def yaml_parser() -> dict:
    parser = argparse.ArgumentParser('Augmnet oarser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])
    parser.add_argument("--local_rank", type=int)
    #parser.add_argument('--var', type=int, default=24)
    #add args.variable here
    args: argparse.Namespace = parser.parse_args()
    args: dict = _parser(args.strings)
    return args

def dict_merge(dct: dict, merge_dct: dict, re=False):
    '''
    Recursive dict merge. Instead updating only top-level keys, dict_merge recuses down into dicts nested
    to an arbitrary depth, updating keys. The ""merge_dct"" is merged into "dct".
    '''
    if merge_dct is None:
        if re:
            return dct
        else:
            return 
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct(k), collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            try:
                dct[k] = type(dct[k])(eval(merge_dct[k])) if type(dct[k]) in (bool, list) else type(dct[k])(
                    merge_dct[k])
            except:
                dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)

##### Timer ######
def now_time():
    time = datetime.datetime.now()
    return str(time)[:19]

# ##### Progress Bar #####

# tqdm_ = partial(tqdm, ncols=125, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

##### Coding #####
def class2one_hot(seg: torch.Tensor, num_class: int) -> torch.Tensor:
    '''
    [b, w, h] containing (0, 1, ..., c) -> [b, c, w, h] containing (0, 1)
    '''
    if len(seg.shape) == 2:
        seg = seg.unsqueeze(dim=0) # Must 3 dim
    if len(seg.shape) == 4:
        seg = seg.squeeze(dim=1)
    assert sset(seg, list(range(num_class))), 'The value of segmentation outside the num_class!'
    b, w, h = seg.shape # Tuple [int, int, int]
    res = torch.stack([seg == c for c in range(num_class)], dim=1).type(torch.int32)
    assert res.shape == (b, num_class, w, h)
    assert one_hot(res)
    
    return res 

def probs2class(probs: torch.Tensor) -> torch.Tensor:
    '''
    [b, c, w, h] containing(float in range(0, 1)) -> [b, w, h] containing ([0, 1, ..., c])
    '''
    b, _, w, h = probs.shape
    assert simplex(probs), '{} is not a probability'.format(probs)
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:
    _, num_class, _, _  = probs.shape
    assert simplex(probs), '{} is not a probability'.format(probs)
    res = class2one_hot(probs2class(probs), num_class)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res

def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label 
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)

def label_onehot_2(inputs, num_class):
    '''
    inputs is class label
    return one_hot label 
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = inputs + 1
    outputs = torch.zeros([batch_size, (num_class + 1), image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)

def uniq(a: torch.Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: torch.Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub) 

def simplex(t: torch.Tensor, axis=1) -> bool:
    '''
    Check if the maticx is the probability in axis dimension.
    '''
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: torch.Tensor, axis=1) ->  bool:
    '''
    Check if the Tensor is One-hot coding
    '''
    return simplex(t, axis) and sset(t, [0, 1])

def intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    a and b must only contain 0 or 1, the function compute the intersection of two tensor.
    a & b
    '''
    assert a.shape == b.shape, '{}.shape must be the same as {}'.format(a, b)
    assert sset(a, [0, 1]), '{} must only contain 0, 1'.format(a)
    assert sset(b, [0, 1]), '{} must only contain 0, 1'.format(b)
    return a & b

class iterator_(object):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        self.dataloader = dcopy(dataloader)
        self.iter_dataloader = iter(dataloader)
        self.cache = None

    def __next__(self):
        try:
            self.cache = self.iter_dataloader.__next__()
            return self.cache
        except StopIteration:
            self.iter_dataloader = iter(self.dataloader)
            self.cache = self.iter_dataloader.__next__()
            return self.cache
    def __cache__(self):
        if self.cache is not None:
            return self.cache
        else:
            warnings.warn('No cache found ,iterator forward')
            return self.__next__()

def apply_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()

##### Scheduler #####
class RampUpScheduler():
    def __init__(self, begin_epoch, max_epoch, max_value, ramp_mult):
        super().__init__()
        self.begin_epoch = begin_epoch
        self.max_epoch = max_epoch
        self.ramp_mult = ramp_mult
        self.max_value = max_value
        self.epoch = 0
    
    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch, self.begin_epoch, self.max_epoch, self.max_value,self.ramp_mult)

    def get_lr(self, epoch, begin_epoch, max_epochs, max_val, mult):
        if epoch < begin_epoch:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1 - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2 )


##### Compute mIoU #####
def mask_label(label, mask):
    '''
    label is the original label (contains -1), mask is the valid region in pseudo label (type=long)
    return a label with invalid region = -1
    '''
    label_tmp = label.clone()
    mask_ = (1 - mask.float()).bool()
    label_tmp[mask_] = -1
    return label_tmp.long()

##### Logger #####
class Logger(object):
    def __init__(self, logFile ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass