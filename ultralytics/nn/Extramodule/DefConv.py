import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        # ¼ÇÂ¼Ô­Ê¼¾í»ýºËÐÎ×´
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        # Ê¹ÓÃ 2D ¾í»ýÉú³ÉÓ³Éä
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        self.bias = None
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        # Éú³ÉÈ¨ÖØ¾ØÕó
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        # Ê¹ÓÃ¾í»ýÓ³Éä¸üÐÂÈ¨ÖØ
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)


class RepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RepConvBlock, self).__init__()
        # ¶¨Òå RepConv Ä£¿é
        self.conv = RepConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=None, groups=1, map_k=3)
        # ÅúÁ¿¹éÒ»»¯²ã
        self.bn = nn.BatchNorm2d(out_channels)
        # ¼¤»îº¯Êý
        self.act = Hswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # H-swish ¼¤»îº¯Êý
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


# ²âÊÔÄ£¿é
if __name__ == "__main__":
    # ´´½¨ RepConvBlock ÊµÀý²¢½øÐÐÇ°Ïò´«²¥²âÊÔ
    block = RepConvBlock(in_channels=3, out_channels=64, stride=1)
    x = torch.randn(1, 3, 224, 224)
    output = block(x)
    print("Output shape:", output.shape)
