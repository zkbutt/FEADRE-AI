import torch
from torch import nn
import torch.nn.functional as F

'''
功能层
'''


class Integral(nn.Module):
    '''
    将按 (reg_max+1)*回归个数 的表示,通过期望求出结果
        先将回归值表示成分布概率: softmax
        再离散化: 通过 reg_max 间隙为1离散数组
        求期望: 通过全连接层

    preg = torch.rand(2, 3, 32)
    integral = Integral(7)
    res = integral(preg)
    print(res.shape)
    print(res)
    '''

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        # 7 生成 tensor([0., 1., 2., 3., 4., 5., 6., 7.]) 的区间
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        '''
        设数据最大为7 形成[0., 1., 2., 3., 4., 5., 6., 7.] 分个点

        :param x: (2, 3, 32=4*8)
        :return:
        '''
        # (2,3,32) -> (2*3*4,8)=[24, 8] 形成该层24个点的分布图
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # [24, 8] ^^ [8,1] -> [24,1]
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x
