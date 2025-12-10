import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm1D(nn.Module):
    """LayerNorm for 1D tensor (B, C, L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, groups=1, act_layer=nn.ReLU):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, state_dim=64):
        super(HSMSSD, self).__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.BCdt_proj = ConvLayer2D(d_model, 3 * state_dim, 1, act_layer=None)
        self.dw = ConvLayer2D(3 * state_dim, 3 * state_dim, 3, padding=1, groups=3 * state_dim, act_layer=None)
        self.hz_proj = ConvLayer2D(d_model, 2 * state_dim, 1, act_layer=None)
        self.out_proj = ConvLayer2D(state_dim, d_model, 1, act_layer=None)

    def forward(self, x):
        BCdt = self.dw(self.BCdt_proj(x))
        B, C, L = BCdt.shape
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = dt.softmax(-1)
        AB = A * B
        h = x @ AB.transpose(-2, -1)
        h, z = torch.split(self.hz_proj(h), [self.state_dim, self.state_dim], dim=1)
        h = self.out_proj(h * F.silu(z) + h)
        return h


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super(FFN, self).__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(EfficientViMBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x, size=None):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        x = self.mixer(self.norm(x.flatten(2)))
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)

        return x


class C2f_EfficientViM(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, ssd_expand=1.0, state_dim=64, mlp_ratio=4.0):
        super(C2f_EfficientViM, self).__init__()
        hidden_channels = int(out_channels * 0.5)
        self.cv1 = ConvLayer2D(in_channels, hidden_channels * 2, 1)
        self.blocks = nn.ModuleList([
            EfficientViMBlock(hidden_channels, ssd_expand=ssd_expand, state_dim=state_dim, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.cv2 = ConvLayer2D(hidden_channels * (num_blocks + 1), out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cv1(x)
        x = torch.chunk(x, 2, dim=1)  # split into two parts
        x1, x2 = x[0], x[1]

        outs = [x1]
        x2 = x2.unsqueeze(-1)  # Adding extra dimension (convert to 4D tensor: [B, C, L, 1])

        for block in self.blocks:
            x2 = block(x2, size=(H, W))
            outs.append(x2.view(B, -1, H, W))

        out = self.cv2(torch.cat(outs, dim=1))
        return out


if __name__ == "__main__":
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量
    x = torch.randn(1, 32, 256, 256).to(device)
    # 初始化 evim 模块
    evim = EfficientViMBlock(dim=32).to(device)
    print(evim)
    # 前向传播
    print("\n微信公众号: AI缝合术!\n")
    output = evim(x)

    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)