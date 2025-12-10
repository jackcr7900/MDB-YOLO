import torch
import torch.nn as nn
import math
import numpy as np

__all__ = ['BiFPN_Concat', 'BiFPN_Concat2']


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class BiFPN_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):  # mutil-layer 1-3 layers ADD or Concat
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x





# BiFPN
# Á½¸öÌØÕ÷Í¼add²Ù×÷
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # ½«È¨ÖØ½øÐÐ¹éÒ»»¯
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
