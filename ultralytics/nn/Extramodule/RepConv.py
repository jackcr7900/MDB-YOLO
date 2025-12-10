import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        # 记录原始卷积核形状
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        # 使用 2D 卷积生成映射
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
        # 生成权重矩阵
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        # 使用卷积映射更新权重
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)


class RepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RepConvBlock, self).__init__()
        # 定义 RepConv 模块
        self.conv = RepConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=None, groups=1, map_k=3)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(out_channels)
        # 激活函数
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
        # H-swish 激活函数
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


# 测试模块
if __name__ == "__main__":
    # 创建 RepConvBlock 实例并进行前向传播测试
    block = RepConvBlock(in_channels=3, out_channels=64, stride=1)
    x = torch.randn(1, 3, 224, 224)
    output = block(x)
    print("Output shape:", output.shape)