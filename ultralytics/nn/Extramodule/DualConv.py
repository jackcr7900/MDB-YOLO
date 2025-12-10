import torch.nn as nn

class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g=2):
        """
        初始化 DualConv 类。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步幅
        :param g: 用于 DualConv 的分组卷积组数
        """
        super(DualConv, self).__init__()
        # 分组卷积
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # 逐点卷积
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        定义 DualConv 如何处理输入图像或输入特征图。
        :param input_data: 输入图像或输入特征图
        :return: 返回输出特征图
        """
        # 同时进行分组卷积和逐点卷积，然后将结果相加
        return self.gc(input_data) + self.pwc(input_data)
