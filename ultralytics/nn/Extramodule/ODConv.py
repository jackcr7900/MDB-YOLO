import torch
import torch.nn as nn
import torch.nn.functional as F

# ×Ô¶¯Ìî³äº¯Êý£¬ÓÃÓÚ¸ù¾Ý¾í»ýºË´óÐ¡×Ô¶¯¼ÆËãÌî³äÁ¿
def autopad(k, p=None, d=1):
    """¸ù¾Ý¾í»ýºË´óÐ¡×Ô¶¯Ìî³ä£¬Ê¹µÃÊä³öÓëÊäÈë³ß´çÒ»ÖÂ¡£"""
    if d > 1:  # ¿¼ÂÇ¾í»ýºËµÄÀ©ÕÅ
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:  # ×Ô¶¯¼ÆËãÌî³äÁ¿
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# ±ê×¼¾í»ýÄ£¿é
class Conv(nn.Module):
    """±ê×¼¾í»ýÄ£¿é£¬°üº¬¾í»ý¡¢Åú¹éÒ»»¯ºÍ¼¤»îº¯Êý¡£"""
    default_act = nn.SiLU()  # Ä¬ÈÏ¼¤»îº¯Êý

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """³õÊ¼»¯¾í»ý²ã£¬°üº¬ÊäÈëÍ¨µÀ¡¢Êä³öÍ¨µÀ¡¢¾í»ýºË´óÐ¡¡¢²½³¤µÈ²ÎÊý¡£"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # Åú¹éÒ»»¯²ã
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Ç°Ïò´«²¥£¬ÒÀ´ÎÓ¦ÓÃ¾í»ý¡¢Åú¹éÒ»»¯ºÍ¼¤»îº¯Êý¡£"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """ÔÚÍÆÀíÊ±Ê¹ÓÃ¾í»ý²Ù×÷£¬Ê¡ÂÔÅú¹éÒ»»¯¡£"""
        return self.act(self.conv(x))

# ODConvµÄ×¢ÒâÁ¦»úÖÆÀà£¬ÓÃÓÚ¼ÆËã²»Í¬Î¬¶ÈÉÏµÄ×¢ÒâÁ¦
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        """³õÊ¼»¯×¢ÒâÁ¦»úÖÆ²ã£¬°üº¬ÊäÈëÍ¨µÀ¡¢Êä³öÍ¨µÀ¡¢¾í»ýºË´óÐ¡¡¢×éÊýµÈ²ÎÊý¡£"""
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # ×ÔÊÊÓ¦È«¾ÖÆ½¾ù³Ø»¯
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)  # È«Á¬½Ó²ã£¬ÓÃÓÚÍ¨µÀËõ¼õ
        self.bn = nn.BatchNorm2d(attention_channel)  # Åú¹éÒ»»¯
        self.relu = nn.ReLU(inplace=True)  # ReLU¼¤»îº¯Êý

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)  # Í¨µÀ×¢ÒâÁ¦²ã
        self.func_channel = self.get_channel_attention  # »ñÈ¡Í¨µÀ×¢ÒâÁ¦

        if in_planes == groups and in_planes == out_planes:  # Éî¶È¾í»ý
            self.func_filter = self.skip  # Èç¹ûÊäÈëÍ¨µÀÊýºÍ×éÊýÏàµÈ£¬ÔòÌø¹ý¹ýÂËÆ÷×¢ÒâÁ¦
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)  # ¹ýÂËÆ÷×¢ÒâÁ¦²ã
            self.func_filter = self.get_filter_attention  # »ñÈ¡¹ýÂËÆ÷×¢ÒâÁ¦

        if kernel_size == 1:  # Èç¹ûÊÇµã¾í»ý£¬Ìø¹ý¿Õ¼ä×¢ÒâÁ¦
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)  # ¿Õ¼ä×¢ÒâÁ¦²ã
            self.func_spatial = self.get_spatial_attention  # »ñÈ¡¿Õ¼ä×¢ÒâÁ¦

        if kernel_num == 1:  # Èç¹ûÖ»ÓÐÒ»¸ö¾í»ýºË£¬Ìø¹ýºË×¢ÒâÁ¦
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)  # ¾í»ýºË×¢ÒâÁ¦²ã
            self.func_kernel = self.get_kernel_attention  # »ñÈ¡¾í»ýºË×¢ÒâÁ¦

        self._initialize_weights()

    def _initialize_weights(self):
        """³õÊ¼»¯È¨ÖØ¡£"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias

, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        """¸üÐÂÎÂ¶È²ÎÊý£¬ÓÃÓÚ¿ØÖÆ×¢ÒâÁ¦»úÖÆµÄÃô¸ÐÐÔ¡£"""
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """Ìø¹ýÄ³¸ö×¢ÒâÁ¦»úÖÆ¡£"""
        return 1.0

    def get_channel_attention(self, x):
        """¼ÆËãÍ¨µÀ×¢ÒâÁ¦¡£"""
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        """¼ÆËã¹ýÂËÆ÷×¢ÒâÁ¦¡£"""
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        """¼ÆËã¿Õ¼ä×¢ÒâÁ¦¡£"""
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        """¼ÆËã¾í»ýºË×¢ÒâÁ¦¡£"""
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        """Ç°Ïò´«²¥£¬ÒÀ´Î¼ÆËãÍ¨µÀ¡¢¹ýÂËÆ÷¡¢¿Õ¼äºÍ¾í»ýºËµÄ×¢ÒâÁ¦¡£"""
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

# ODConv2dÀàÊµÏÖ£¬¼¯³É×¢ÒâÁ¦»úÖÆµÄ¾í»ý²ã
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, reduction=0.0625, kernel_num=4):
        """³õÊ¼»¯ODConv¾í»ý²ã£¬°üº¬ÊäÈëÍ¨µÀ¡¢Êä³öÍ¨µÀ¡¢¾í»ýºË´óÐ¡¡¢²½³¤¡¢Ìî³äµÈ²ÎÊý¡£"""
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        """³õÊ¼»¯È¨ÖØ¡£"""
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        """¸üÐÂÎÂ¶È²ÎÊý¡£"""
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        """³£¹æÇ°Ïò´«²¥ÊµÏÖ£¬°üº¬¶à¸ö¾í»ýºËµÄ¾í»ý²Ù×÷¡£"""
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        """µ±¾í»ýºË´óÐ¡Îª1x1Ê±µÄÇ°Ïò´«²¥ÊµÏÖ¡£"""
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        """µ÷ÓÃÇ°Ïò´«²¥·½·¨¡£"""
        return self._forward_impl(x)

class ODConv2d_yolo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, k=1)
        self.dcnv3 = ODConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                               dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dcnv3(x)
        x = self.gelu(self.bn(x))
        return x


