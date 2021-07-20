import torch.nn as nn
import torch.nn.functional as F


class FPN_out4add(nn.Module):

    def __init__(self, in_channels_list, out_channel=256,
                 use_p5=True, num_outs=5, is_init=False, use_back_conv=True):
        '''

        :param in_channels_list: 写死只支持3层
        :param out_channel:
        :param num_outs: 只支持3或5
        '''
        super(FPN_out4add, self).__init__()
        self.add_module('prj_5', nn.Conv2d(in_channels_list[2], out_channel, kernel_size=1))
        # self.prj_5 = nn.Conv2d(in_channels_list[2], out_channel, kernel_size=1)
        self.prj_4 = nn.Conv2d(in_channels_list[1], out_channel, kernel_size=1)
        self.prj_3 = nn.Conv2d(in_channels_list[0], out_channel, kernel_size=1)

        self.use_back_conv = use_back_conv
        if self.use_back_conv:  # 尺寸不变
            self.conv_5 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
            self.conv_4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
            self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if num_outs == 5:
            # 输出5层才启动
            if use_p5:
                self.conv_out6 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)
            else:
                self.conv_out6 = nn.Conv2d(in_channels_list[2], out_channel, kernel_size=3, padding=1, stride=2)
            self.conv_out7 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)

        if is_init:
            self.init_conv(self)

        self.use_p5 = use_p5
        self.num_outs = num_outs
        # self.apply(finit_conv_kaiming)
        self.out_channel = out_channel

    def init_conv(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def upsamplelike(self, inputs):
        '''

        :param inputs: 第二个是尺寸 size_hw 行列
        :return:
        '''
        src, target = inputs
        #  nn.Upsample 与 F.interpolate 等价
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        if self.use_back_conv:
            P3 = self.conv_3(P3)
            P4 = self.conv_4(P4)
            P5 = self.conv_5(P5)

        if self.num_outs == 5:
            P5 = P5 if self.use_p5 else C5
            P6 = self.conv_out6(P5)
            P7 = self.conv_out7(F.relu(P6))
            return [P3, P4, P5, P6, P7]
        else:
            return [P3, P4, P5]


class PAN_out4add(FPN_out4add):

    def __init__(self, in_channels_list, out_channel=256,
                 use_p5=True, num_outs=5, is_init=False, use_back_conv=True):
        '''

        :param in_channels_list: 写死只支持3层
        :param out_channel:
        :param num_outs: 只支持3或5
        '''
        super().__init__(in_channels_list, out_channel, use_p5, num_outs, is_init, use_back_conv)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P4 = P4 + self.upsamplelike([P5, C4])
        P5 = P5 + self.upsamplelike([P4, C5])

        if self.use_back_conv:
            P3 = self.conv_3(P3)
            P4 = self.conv_4(P4)
            P5 = self.conv_5(P5)

        if self.num_outs == 5:
            P5 = P5 if self.use_p5 else C5
            P6 = self.conv_out6(P5)
            P7 = self.conv_out7(F.relu(P6))
            return [P3, P4, P5, P6, P7]
        else:
            return [P3, P4, P5]
