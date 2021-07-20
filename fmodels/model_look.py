import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

import torch
import torchvision.models as models
from torch import nn
from thop import profile


def f_look_tw(model, input=(1, 3, 416, 416), name='model_look'):
    import tensorwatch as tw
    # 用这个即可---查看网络的统计结果---
    args_pd = tw.model_stats(model, input)
    args_pd.to_excel(name + '.xlsx')
    script_models = torch.jit.trace(model, input)
    script_models.save(name + '_jit' + '.pt')
    print('文件生成成功 %s', name + '.xlsx')


def f_look_summary(model, input=(3, 416, 416), device="cpu"):
    '''
    这个不支持 tuple 输入  没有.size()属性会报错
    :param model:
    :param input:
    :param device:
    :return:
    '''
    from torchsummary import summary
    if not isinstance(input, tuple):
        input = tuple(input)
    summary1 = summary(model, input, device=device)
    print(type(summary1))


def f_calc_flops_params(model, size_hw):
    inputs = torch.randn(1, 3, *size_hw)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)
    return flops, params


if __name__ == '__main__':
    '''
    '''
    # data_inputs_list = [1, 3, 640, 640]
    data_inputs_list = [1, 3, 416, 416]

    torch.random.manual_seed(20201025)  # 3746401707500
    data_inputs_ts = torch.rand(data_inputs_list, dtype=torch.float)

    # model = models.AlexNet()
    # model = models.densenet161(pretrained=True)  # 能力 22.35  6.20  ---top2
    # model = FRebuild4densenet161(model, None)
    # return_layers = {'layer1': 1, 'layer2': 2, 'layer3': 3}

    # model = models.wide_resnet50_2(pretrained=True)  # 能力 21.49 5.91  ---top1
    # model = models.resnext50_32x4d(pretrained=True)  # 能力 22.38 6.30 ---top3
    # model = models.mobilenet_v2(pretrained=True)  # 能力 28.12 9.71 ---速度top1
    # model = models.mnasnet1_0(pretrained=True)
    # model = models.shufflenet_v2_x1_0(pretrained=True)

    # model = models.resnet50(pretrained=True)  # 下采样倍数32 能力23.85 7.13
    model = models.resnet34(pretrained=True)  # 下采样倍数32 能力23.85 7.13
    # return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    # my_model = FModelOne2More(model, return_layers)
    # data_outputs = my_model(data_inputs_ts)
    # for k, v in data_outputs.items():
    #     print(v.shape)

    f_calc_flops_params(model, (300, 300))

    # f替换(model)

    # f_look_tw(model, data_inputs_list, name='f_look_tw')
    # f_look_summary(model, data_inputs_list)

    # model = darknet53()
    # modelsize(model, torch.rand(1, 3, 416, 416))
    '''
    torch.Size([1, 512, 80, 80])
    torch.Size([1, 1024, 40, 40])
    torch.Size([1, 2048, 20, 20])
    '''

    # model = MobileNetV1()  # 下采样倍数32 能力23.85 7.13
    # model = models.squeezenet1_0(pretrained=True)
    # model = models.vgg.vgg16(pretrained=True)
    # model = models.shufflenet_v2_x1_0(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.mnasnet1_0(pretrained=True)  # 能力 26.49 8.456
    # model = models.inception_v3(pretrained=True)  # 能力 22.55 6.44

    '''-----------------模型分析 开始-----------------------'''

    # 用这个即可---查看网络的统计结果---
    # args_pd = tw.model_stats(model, data_inputs_list)
    # args_pd.to_excel('model_log.xlsx')

    # # print(type(args_pd))
    # print(args_pd)

    # from torchsummary import summary

    # summary1 = summary(model, (3, 640, 640))
    # print(type(summary1))

    # other()
    '''-----------------模型分析 完成-----------------------'''

    print('---%s--main执行完成------ ', os.path.basename(__file__))
