# https://github.com/sunsmarterjie/yolov12

import math

import numpy as np
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape — supports int or tuple kernel sizes"""
    if isinstance(k, int):
        # scalar kernel
        if d > 1:
            k = d * (k - 1) + 1
        return k // 2 if p is None else p
    elif isinstance(k, tuple):
        # 2D kernel like (3,3)
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        return tuple(x // 2 for x in k) if p is None else p
    else:
        raise TypeError(f"Invalid kernel type: {type(k)}")



#NOTE From https://github.com/sunsmarterjie/yolov12/blob/main/ultralytics/nn/modules/conv.py
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))