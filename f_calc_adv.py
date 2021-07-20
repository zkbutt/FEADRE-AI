import torch
import numpy as np


def f_cre_grid_cells(size, stride=1, is_center=False,
                     is_flatten=True, is_swap=True, num_repeat=1,
                     device=torch.device('cpu')
                     ):
    '''

    :param size: 决定遍历顺序 第一维fix先固定   第二维b先遍历
    :param stride: 这个为1 为size大小的索引
    :param is_center:  网点是否居中
        off_xy 偏移量 通常为 0~1
    :param is_flatten: 通常这个不要动 因为最后一句拉平 所有无效
    :param is_swap: 是否交换size顺序
    :param num_repeat: 单体复制数 用于批量匹配anc
    :return:
        例 [10, 15] 表示row10(y) col15(x)
            用于遍历矩阵index 采用 is_swap =False 表示先列后行 [[0,0],[0,1]...]
            用于坐标 is_swap =True 表示xy点遍历 [[0,0],[1,0]...]
        [150, 2]
    '''
    fix, b = size  # fix 为先固定  b先遍历
    if is_center:
        off_xy = 0.5
    else:
        off_xy = 0
    x_b_ = (torch.arange(b, dtype=torch.float, device=device) + off_xy) * stride
    y_fix_ = (torch.arange(fix, dtype=torch.float, device=device) + off_xy) * stride
    y_fix, x_b = torch.meshgrid(y_fix_, x_b_)  # 第一个固定
    if is_flatten:
        y_fix = y_fix.flatten()  # torch.Size([10, 15])
        x_b = x_b.flatten()  # torch.Size([10, 15])
    if is_swap:
        stack = torch.stack((x_b, y_fix), dim=-1)
    else:
        stack = torch.stack((y_fix, x_b), dim=-1)

    # 这个直接拉平
    stack = torch.repeat_interleave(stack.view(-1, 2), num_repeat, dim=0)
    return stack
    # return y, x


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


if __name__ == '__main__':
    index_rowcol = f_cre_grid_cells([10, 15], stride=1, is_center=False, is_swap=True,
                                    is_flatten=True, num_repeat=3)
    print(index_rowcol)
