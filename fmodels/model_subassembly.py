import torch
import torch.nn as nn

from tools.fmodels.f_fun_activation import Mish, SiLU

'''
模型组件
'''


def conv_same(k):
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class AffineChannel(torch.nn.Module):
    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, ))
        self.bias = torch.nn.Parameter(torch.randn(num_features, ))

    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        transpose_x = x.permute(0, 2, 3, 1)
        flatten_x = transpose_x.reshape(N * H * W, C)
        out = flatten_x * self.weight + self.bias
        out = out.reshape(N, H, W, C)
        out = out.permute(0, 3, 1, 2)
        return out


class FConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 k,  # kernel_size
                 s=1,  # stride
                 p=0,  # padding 等于 None 是自动same
                 d=1,  # dilation空洞
                 g=1,  # g groups 一般不动
                 is_bias=True,
                 norm='bn',  # None bn,gn,af
                 act='leaky',  # None relu leaky mish silu identity
                 is_freeze=False,
                 use_dcn=False):
        '''
        有 bn 建议不要 bias
        # 普通conv

        # 降维
        '''
        super(FConv2d, self).__init__()
        self.groups = g
        self.act = act
        self.is_freeze = is_freeze
        self.use_dcn = use_dcn

        if p is None:
            # conv默认为0
            p = conv_same(k)

        # conv
        if use_dcn:
            pass
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s,
                                        padding=p, dilation=d, bias=is_bias)
        # 正则化方式 normalization
        if norm == 'bn':
            self.normalization = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.normalization = torch.nn.GroupNorm(num_groups=g, num_channels=out_channels)
        elif norm == 'af':
            self.normalization = AffineChannel(out_channels)
        else:
            self.normalization = None

        # act
        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1, inplace=True)
        elif act == 'mish':
            self.act = Mish()
        elif act == 'silu':
            self.act = SiLU()
        elif act == 'identity':
            self.act = nn.Identity()
        else:
            self.act = None

        self.name_act = act

        if is_freeze:  # 一般不锁定
            self.freeze()

    def freeze(self):
        # 冻结
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
        if self.bn is not None:
            self.bn.weight.requires_grad = False
            self.bn.bias.requires_grad = False
        if self.gn is not None:
            self.gn.weight.requires_grad = False
            self.gn.bias.requires_grad = False
        if self.act is not None:
            self.act.weight.requires_grad = False
            self.act.bias.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channel, out_channel, 3, padding=1, groups=in_channel, bias=False)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.dwnorm = nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pwnorm = nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dwnorm(x)
        x = self.pwnorm(x)
        x = self.act(x)
        return x
