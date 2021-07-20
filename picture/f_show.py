import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from tools.datas.f_data_pretreatment4np import f_recover_normalization4ts

'''
plt 常用颜色: 'lightgreen' 'red' 'tan'
'''


def _draw_box4plt(boxes, color='red', ax=None, recover_sizewh=None):
    '''

    :param boxes:
    :param color:
    :param ax:  ax = plt.gca()
    :return:
    '''
    if recover_sizewh is not None:
        whwh = np.tile(np.array(recover_sizewh), 2)  # 整体复制 tile
        boxes = boxes * whwh
    for i, box in enumerate(boxes):
        l, t, r, b = box
        w = r - l
        h = b - t
        if ax is None:
            plt.Rectangle((l, t), w, h, color=color, fill=False, linewidth=1)
        else:
            ax.add_patch(plt.Rectangle((l, t), w, h, color=color, fill=False, linewidth=1))
        x = l + w / 2
        y = t + h / 2
        plt.scatter(x, y, marker='x', color=color, s=40, label='First')


def f_show_od_ts4plt_v3(img_ts, g_ltrb=None, p_ltrb=None, is_recover_size=False,
                        glabels_text=None, plabels_text=None, p_scores_float=None, grids=None,
                        is_normal=False):
    assert isinstance(img_ts, torch.Tensor)
    # c,h,w -> h,w,c
    if is_normal:
        img_ts = img_ts.clone()
        img_ts = f_recover_normalization4ts(img_ts)
    img_np_rgb = img_ts.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
    f_show_od_np4plt_v3(img_np_rgb, g_ltrb=g_ltrb, p_ltrb=p_ltrb,
                        is_recover_size=is_recover_size,
                        glabels_text=glabels_text,
                        plabels_text=plabels_text,
                        p_scores_float=p_scores_float,
                        grid_wh_np=grids)


def f_show_od_np4plt_v3(img_np, g_ltrb=None, p_ltrb=None, other_ltrb=None, is_recover_size=False,
                        glabels_text=None, plabels_text=None,
                        p_scores_float=None, grid_wh_np=None):
    img_np = _convert_uint8(img_np)
    plt.imshow(img_np)
    # plt.show()
    ax = plt.gca()

    if is_recover_size:
        recover_sizewh = img_np.shape[:2][::-1]  # npwh
    else:
        recover_sizewh = None

    if p_ltrb is not None:
        plt.title(
            '%s x %s (num_pos = %s)' % (str(img_np.shape[1]), str(img_np.shape[0]), str(len(p_ltrb))))
    else:
        plt.title('%s x %s ' % (str(img_np.shape[1]), str(img_np.shape[0])))

    if other_ltrb is not None:
        _draw_box4plt(other_ltrb, color='tan', ax=ax, recover_sizewh=recover_sizewh)

    if g_ltrb is not None:
        _draw_box4plt(g_ltrb, color='lightgreen', ax=ax, recover_sizewh=recover_sizewh)
        # ax.text(100, 200, "label 0.9", bbox={'facecolor':'blue', 'alpha':0.5})

    if p_ltrb is not None:
        _draw_box4plt(p_ltrb, color='red', ax=ax, recover_sizewh=recover_sizewh)

    plt.show()


# ********************   新方法在上面

def keypoint_painter(images, maps, img_h, img_w, numpy_array=False,
                     phase_gt=False, center_map=None):
    images = images.clone().cpu().data.numpy().transpose([0, 2, 3, 1])
    maps = maps.clone().cpu().data.numpy()
    if center_map is not None:
        center_map = center_map.clone().cpu().data.numpy()
    imgs_tensor = []
    if phase_gt:
        for img, map, c_map in zip(images, maps, center_map):
            img = cv2.resize(img, (img_w, img_h))
            for m in map[:14]:
                h, w = np.unravel_index(m.argmax(), m.shape)
                x = int(w * img_w / m.shape[1])
                y = int(h * img_h / m.shape[0])
                img = cv2.circle(img.copy(), (x, y), radius=1, thickness=2, color=(255, 0, 0))
            h, w = np.unravel_index(c_map.argmax(), c_map.shape)
            x = int(w * img_w / c_map.shape[1])
            y = int(h * img_h / c_map.shape[0])
            img = cv2.circle(img.copy(), (x, y), radius=1, thickness=2, color=(0, 0, 255))
            if numpy_array:
                imgs_tensor.append(img.astype(np.uint8))
            else:
                imgs_tensor.append(transforms.ToTensor()(img))
    else:

        for img, map_6 in zip(images, maps):
            img = cv2.resize(img, (img_w, img_h))
            for step_map in map_6:
                img_copy = img.copy()
                for m in step_map[:14]:
                    h, w = np.unravel_index(m.argmax(), m.shape)
                    x = int(w * img_w / m.shape[1])
                    y = int(h * img_h / m.shape[0])
                    img_copy = cv2.circle(img_copy.copy(), (x, y), radius=1, thickness=2, color=(255, 0, 0))
                if numpy_array:
                    imgs_tensor.append(img_copy.astype(np.uint8))
                else:
                    imgs_tensor.append(transforms.ToTensor()(img_copy))
    return imgs_tensor


def _convert_uint8(img_np):
    if img_np.dtype is not np.uint8:
        img_np_uint8 = img_np.copy()
        img_np_uint8 = img_np_uint8.astype(np.uint8)
    else:
        return img_np
    return img_np_uint8


def f_show_pic_np4plt(img_np):
    '''
    不支持float32
    :param pic:
    :return:
    '''
    img_np = _convert_uint8(img_np)
    plt.imshow(img_np)
    plt.show()


def f_show_pic_np4cv(img_np):
    img_np = _convert_uint8(img_np)
    cv2.imshow('Example', img_np)
    cv2.waitKey(0)


def f_show_kp_np4plt(img_np, boxes_ltrb_input, kps_xy_input, mask_kps, g_ltrb=None,
                     plabels_text=None, p_scores_float=None,
                     is_recover_size=True):
    '''

    :param img_np:
    :param boxes_ltrb_input:
    :param kps_xy_input:
    :param mask_kps:  预测时自动生成一个  torch.ones_like(p_keypoints_f, dtype=torch.bool)
    :param g_ltrb:
    :param plabels_text: [类型名称,]
    :param p_scores_float:[对应分数的list]
    :param is_recover_size:
    :return:
    '''
    img_np = _convert_uint8(img_np)

    assert isinstance(img_np, np.ndarray)
    # import matplotlib.pyplot as plt
    plt.title('%s x %s (num_pos = %s)' % (str(img_np.shape[1]), str(img_np.shape[0]), str(len(boxes_ltrb_input))))
    # plt.imshow(img_np)
    # plt.show()
    whwh = np.tile(np.array(img_np.shape[:2][::-1]), 2)  # 整体复制
    # plt.figure(whwh)  #要报错
    plt.imshow(img_np, alpha=0.7)
    current_axis = plt.gca()
    if g_ltrb is not None:
        if is_recover_size:
            g_ltrb = g_ltrb * whwh
        for box in g_ltrb:
            l, t, r, b = box
            plt_rectangle = plt.Rectangle((l, t), r - l, b - t, color='lightcyan', fill=False, linewidth=3)
            current_axis.add_patch(plt_rectangle)
            # x, y = c[:2]
            # r = 4
            # # 空心
            # # draw.arc((x - r, y - r, x + r, y + r), 0, 360, fill=color)
            # # 实心
            # draw.chord((x - r, y - r, x + r, y + r), 0, 360, fill=_color)

    for i, box_ltrb_input in enumerate(boxes_ltrb_input):
        l, t, r, b = box_ltrb_input
        # ltwh
        current_axis.add_patch(plt.Rectangle((l, t), r - l, b - t, color='green', fill=False, linewidth=2))
        _xys = kps_xy_input[i][mask_kps[i]]

        ''' 就这里不一样 与f_plt_od_np'''
        # 这个是多点连线 加参数后 marker='o' 失效,变成多点连线
        plt.scatter(_xys[::2], _xys[1::2],
                    color='r', s=3, alpha=0.5)
        # plt.plot(_xys[::2], _xys[1::2],
        #          marker='o',
        #          markerfacecolor='red',  # 点颜⾊
        #          markersize=3,  # 点⼤⼩
        #          markeredgecolor='green',  # 点边缘颜⾊
        #          markeredgewidth=2,  # 点边缘宽度
        #          )

        if plabels_text is not None:
            # labels : tensor -> int
            show_text = text = "{} : {:.3f}".format(plabels_text[i], p_scores_float[i])
            current_axis.text(l + 8, t - 10, show_text, size=8, color='white',
                              bbox={'facecolor': 'blue', 'alpha': 0.1})
    plt.show()


def f_show_cpm4input(img_np, kps_xy_input, heatmap_center_input):
    '''

    :param img_np:
    :param kps_xy_input: (14, 2)
    :param heatmap_center_input: (368, 368)
    :return:
    '''
    img_np = _convert_uint8(img_np)
    plt.imshow(img_np, alpha=0.7)
    for xy in kps_xy_input:
        plt.scatter(xy[0], xy[1], color='r', s=5, alpha=0.5)
    plt.imshow(heatmap_center_input, alpha=0.5)
    plt.show()


def f_show_cpm4t_all(img_np, kps_xy_input, heatmap_center_input, heatmap_t, size_wh_input):
    '''
    弄到input进行处理
    :param img_np:
    :param kps_xy_t: (14, 2)
    :param heatmap_center_input: (368, 368,1)
    :param heatmap_t: (46, 46, 14)
    :param size_wh_t: (46, 46, 15)
    :return:
    '''
    img_np = _convert_uint8(img_np)
    plt.imshow(img_np, alpha=1.0)
    for xy in kps_xy_input:
        plt.scatter(xy[0], xy[1], color='r', s=5, alpha=0.5)
    plt.imshow(heatmap_center_input, alpha=0.3)

    h, w, c = heatmap_t.shape
    img_heatmap_z = np.zeros((size_wh_input[1], size_wh_input[0], 1), dtype=np.uint8)
    for i in range(c):
        img_heatmap = heatmap_t[..., i][..., None]
        img_heatmap = cv2.resize(img_heatmap, size_wh_input)  # wh
        # img_heatmap = _convert_uint8(img_heatmap)
        img_heatmap_z = np.maximum(img_heatmap_z, img_heatmap[..., None])

    plt.imshow(img_heatmap_z, alpha=0.3)
    plt.show()
