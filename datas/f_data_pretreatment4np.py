import math
import os

import torch
import cv2
import numpy as np
import types
from numpy import random

from tools.GLOBAL_LOG import flog


class BasePretreatment:

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg


def cre_transform_np4train(cfg):
    if cfg.tcfg_pic_handler is not None and (
            cfg.tcfg_pic_handler == 'widerface'
            or cfg.tcfg_pic_handler == 'face98'
    ):
        flog.warning('特殊数据处理 不要缩小 %s', )
        data_transform = {
            "train": Compose([
                Uint2Float32_np(),  # image int8转换成float [0,256)
                PhotometricDistort(),  # 图片处理集合
                # Expand(cfg.PIC_MEAN, cfg.NUM_KEYPOINTS),  # 放大缩小图片 只会缩小
                FRandomSampleCrop(cfg.NUM_KEYPOINTS),  # 随机剪切定位 keypoints 只会放大
                FRandomMirror(cfg.NAME_DATA),
                FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
                Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                ConvertColor(current='BGR', transform='RGB'),
                img_np_bgr2ts_rgb(is_box_oned=False),
            ], cfg)
        }
    elif cfg.MODE_VIS == 'keypoints':
        data_transform = {
            "train": Compose([
                Uint2Float32_np(),  # image int8转换成float [0,256)
                PhotometricDistort(),  # 图片处理集合
                FExpand(cfg.PIC_MEAN, cfg.NUM_KEYPOINTS),  # 放大缩小图片 只会缩小
                FRandomSampleCrop(cfg.NUM_KEYPOINTS),  # 随机剪切定位 keypoints 只会放大
                FRandomMirror(cfg.NAME_DATA),
                FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
                Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                ConvertColor(current='BGR', transform='RGB'),
                img_np_bgr2ts_rgb(is_box_oned=False),
            ], cfg)
        }
    else:
        if cfg.USE_BASE4NP:
            flog.error('使用的是 USE_BASE4NP 模式 %s', cfg.USE_BASE4NP)
            data_transform = {
                "train": Compose([
                    Uint2Float32_np(),  # image int8转换成float [0,256)
                    FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
                    Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                    ConvertColor(current='BGR', transform='RGB'),
                    img_np_bgr2ts_rgb(is_box_oned=False),
                ], cfg)
            }
        else:
            data_transform = {
                "train": Compose([
                    Uint2Float32_np(),  # image int8转换成float [0,256)
                    # ToAbsoluteCoords(),  # 输入已是原图不需要恢复 boxes 恢复原图尺寸
                    PhotometricDistort(),  # 图片处理集合
                    FExpand(cfg.PIC_MEAN),  # 放大缩小图片 无需cfg.NUM_KEYPOINTS
                    FRandomSampleCrop(),  # 随机剪切定位 无需cfg.NUM_KEYPOINTS
                    FRandomMirror(),  # 无需cfg.NAME_DATA
                    # ToPercentCoords(),  # boxes 按原图归一化 最后统一归一 最后ToTensor 统一归一
                    FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
                    Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
                    ConvertColor(current='BGR', transform='RGB'),
                    img_np_bgr2ts_rgb(is_box_oned=False),
                ], cfg)
            }

    data_transform["val"] = Compose([
        FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
        Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
        ConvertColor(current='BGR', transform='RGB'),
        img_np_bgr2ts_rgb(is_box_oned=False),
    ], cfg)

    return data_transform


def cre_transform_balance_data(cfg):
    data_transform = {
        "train": Compose([
            Uint2Float32_np(),  # image int8转换成float [0,256)
            # --------------- 图形整体 ------------
            RandomContrast(),  # 随机透明度
            ConvertColor(transform='HSV'),  # bgr -> hsv
            RandomSaturation(),  # 随机色彩'
            RandomHue(),  # HUE变化
            ConvertColor(current='HSV', transform='BGR'),  # hsv -> bgr
            RandomContrast(),  # 随机透明度
            RandomBrightness(),  # 随机亮度增强
            RandomGray(),  # 灰度
            FRandomNoise(),  # 杂点
            Ffilter_gaussian(),  # 高斯模糊
            Fgamma_transform(),  # Gamma 暗加强

            # --------------- 变换 ------------
            FExpand(cfg.PIC_MEAN, cfg.NUM_KEYPOINTS),  # 放大缩小图片 只会缩小
            FRandomSampleCrop(cfg.NUM_KEYPOINTS),  # 随机剪切定位 keypoints 只会放大
            FRandomMirror(cfg.NAME_DATA),  # 水平镜像
            FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),

            # --------------- 后处理 ------------
            Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
            ConvertColor(current='BGR', transform='RGB'),
            img_np_bgr2ts_rgb(is_box_oned=False),
        ], cfg)
    }
    data_transform["val"] = Compose([
        FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
        Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
        ConvertColor(current='BGR', transform='RGB'),
        img_np_bgr2ts_rgb(is_box_oned=False),
    ], cfg)

    return data_transform


def cre_transform_np4test(cfg):
    data_transform = Compose([
        FPResize(cfg.P_SIZE, cfg.IS_SIZE_SCOPE),  # 针对一张图片
        Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
        ConvertColor(current='BGR', transform='RGB'),
        img_np_bgr2ts_rgb(is_box_oned=False),
    ], cfg)
    return data_transform


def cre_transform_np4base(cfg):
    flog.warning('预处理使用 cre_transform_base4np', )
    data_transform = {
        "train": Compose([
            FResize(cfg.IMAGE_SIZE, cfg.IS_MULTI_SCALE_V2, cfg.MULTI_SCALE_VAL_V2),
            Normalize(),
            ConvertColor(current='BGR', transform='RGB'),
            img_np_bgr2ts_rgb(is_box_oned=False),
        ], cfg)
    }

    # data_transform["val"] = Compose([
    #     FPResize(cfg.P_SIZE, cfg.P_SIZE),
    #     Normalize(cfg.PIC_MEAN, cfg.PIC_STD),
    #     ConvertColor(current='BGR', transform='RGB'),
    #     img_np_bgr2ts_rgb(is_box_oned=False),
    # ], cfg)

    return data_transform


def _copy_box(boxes):
    if isinstance(boxes, np.ndarray):
        boxes_ = boxes.copy()
    elif isinstance(boxes, torch.Tensor):
        boxes_ = boxes.clone()
    else:
        raise Exception('类型错误', type(boxes))

    return boxes_


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(BasePretreatment):

    def __init__(self, transforms, cfg=None):
        super(Compose, self).__init__(cfg)
        self.transforms = transforms

    def __call__(self, img, target):
        # f_plt_show_cv(image,boxes)
        for t in self.transforms:
            img, target = t(img, target)
            if target is not None:
                if 'boxes' in target and 'labels' in target:
                    if len(target['boxes']) != len(target['labels']):
                        flog.warning('!!! 数据有问题 Compose  %s %s %s ', len(target['boxes']), len(target['labels']), t)
        return img, target


class Uint2Float32_np(object):
    def __call__(self, image, target):
        '''cv打开的np 默认是uint8'''
        return image.astype(np.float32), target


class ts2img_np_bgr(object):
    def __call__(self, tensor, target):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), target


# --------------- 图形整体 ------------

class FRandomNoise:
    '''
    随机噪声 噪点
    :param image:
    :param threshold:
    :return:
    '''

    def __init__(self, threshold=32) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(self, img_np, target):
        if random.randint(2):
            noise = np.random.uniform(low=-1, high=1, size=img_np.shape)
            img_np = img_np + noise * self.threshold
            img_np = np.clip(img_np, 0, 255)

        return img_np, target


class Ffilter_gaussian:
    '''
    高斯滤波器是根据高斯函数的形状来选择权值的线性平滑滤波器，滤波器符合二维高斯分布
    模糊
    :param image:
    :param threshold:
    :return:
    '''

    def __init__(self, k_size=5) -> None:
        super().__init__()
        self.k_size = k_size

    def __call__(self, img_np, target):
        if random.randint(2):
            img_np = cv2.GaussianBlur(img_np, (self.k_size, self.k_size), 3)
        return img_np, target


class Fgamma_transform:
    '''
    Gamma变换就是用来图像增强，通过非线性变换提升了暗部细节
    :param image:
    :param threshold:
    :return:
    '''

    def __init__(self, gamma=1.6) -> None:
        super().__init__()
        self.gamma = gamma

    def __call__(self, img_np, target):
        if random.randint(2):
            max_value = np.max(img_np)
            min_value = np.min(img_np)
            value_l = max_value - min_value
            img_np = (img_np - min_value) / value_l
            img_np = np.power(img_np, self.gamma)
            img_np = img_np * 255
        return img_np, target


class RandomSaturation(object):
    '''随机色彩 需要HSV'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomContrast(object):
    '''随机透明度'''

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target):
        if random.randint(2):  # 50%
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        # f_plt_show_cv(image)
        return image, target


class RandomGray:
    '''随机灰度'''

    def __call__(self, image, target):
        if random.random() > 0.9:
            _img = np.zeros(image.shape, dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _img[:, :, 0] = gray
            _img[:, :, 1] = gray
            _img[:, :, 2] = gray
            image = _img

        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        '''随机亮度增强'''
        if random.randint(2):  # 50%
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        # f_plt_show_cv(image)
        return image, target


# --------------- 变换 ------------
class FRandomRotate(object):
    '''
    有框不能用旋转
    '''

    def __init__(self, max_degree=20):
        self.max_degree = max_degree

    def __call__(self, img_np, target):
        degree = random.uniform(-self.max_degree, self.max_degree)
        h, w, c = img_np.shape
        cx, cy = w / 2.0, h / 2.0

        ''' 图片处理 '''
        # mat rotate 1 center 2 angle 3 缩放系数
        matRotate = cv2.getRotationMatrix2D((cy, cx), degree, 1.0)
        img_np = cv2.warpAffine(img_np, matRotate, (h, w))

        if 'boxes' in target:
            boxes_ltrb_ts = target['boxes']
            a = -degree / 180.0 * math.pi
            # boxes = torch.from_numpy(boxes)
            new_boxes_ltrb_ts = torch.zeros_like(boxes_ltrb_ts)
            new_boxes_ltrb_ts[:, 0] = boxes_ltrb_ts[:, 1]
            new_boxes_ltrb_ts[:, 1] = boxes_ltrb_ts[:, 0]
            new_boxes_ltrb_ts[:, 2] = boxes_ltrb_ts[:, 3]
            new_boxes_ltrb_ts[:, 3] = boxes_ltrb_ts[:, 2]
            for i in range(boxes_ltrb_ts.shape[0]):
                ymin, xmin, ymax, xmax = new_boxes_ltrb_ts[i, :]
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                x0, y0 = xmin, ymin
                x1, y1 = xmin, ymax
                x2, y2 = xmax, ymin
                x3, y3 = xmax, ymax
                z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
                tp = torch.zeros_like(z)
                tp[:, 1] = (z[:, 1] - cx) * math.cos(a) - (z[:, 0] - cy) * math.sin(a) + cx
                tp[:, 0] = (z[:, 1] - cx) * math.sin(a) + (z[:, 0] - cy) * math.cos(a) + cy
                ymax, xmax = torch.max(tp, dim=0)[0]
                ymin, xmin = torch.min(tp, dim=0)[0]
                new_boxes_ltrb_ts[i] = torch.stack([ymin, xmin, ymax, xmax])
            new_boxes_ltrb_ts[:, 1::2].clamp_(min=0, max=h - 1)
            new_boxes_ltrb_ts[:, 0::2].clamp_(min=0, max=w - 1)
            boxes_ltrb_ts[:, 0] = new_boxes_ltrb_ts[:, 1]
            boxes_ltrb_ts[:, 1] = new_boxes_ltrb_ts[:, 0]
            boxes_ltrb_ts[:, 2] = new_boxes_ltrb_ts[:, 3]
            boxes_ltrb_ts[:, 3] = new_boxes_ltrb_ts[:, 2]
            target['boxes'] = boxes_ltrb_ts

        if 'keypoints' in target:
            raise Exception('关键点检测不在这里做')
        return img_np, target


class FExpand(object):
    '''随机 缩小放在图片中某处  其它部份为黑色'''

    def __init__(self, mean, num_keypoints=None):
        ''' 均值用于扩展边界 '''
        self.mean = mean
        self.num_keypoints = num_keypoints

    def __call__(self, image, target):
        if random.randint(2):
            return image, target

        height, width, depth = image.shape
        ratio = random.uniform(1, 3)  # 缩小 1~4的一个比例
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image
        # 原尺寸不要了
        boxes = _copy_box(target['boxes'])

        if isinstance(boxes, np.ndarray):
            boxes[:, :2] += (int(left), int(top))  # lt
            boxes[:, 2:] += (int(left), int(top))  # rb
        elif isinstance(boxes, torch.Tensor):
            boxes[:, :2] += torch.tensor((int(left), int(top)))
            boxes[:, 2:] += torch.tensor((int(left), int(top)))
        else:
            raise Exception('target[boxes]类型错误', type(boxes))

        if 'keypoints' in target:
            assert self.num_keypoints > 0, 'cfg.NUM_KEYPOINTS = %s 设置错误' % self.num_keypoints
            if isinstance(boxes, np.ndarray):
                target['keypoints'] += (int(left), int(top)) * self.num_keypoints
            elif isinstance(boxes, torch.Tensor):
                target['keypoints'] += torch.tensor((int(left), int(top)) * self.num_keypoints)
            else:
                raise Exception('target[keypoints]类型错误', type(target['keypoints']))

        target['boxes'] = boxes
        # f_plt_show_cv(image,torch.tensor(boxes))
        return image, target


class FRandomSampleCrop(object):
    """
    随机Crop
    """

    def __init__(self, num_keypoints=None):
        # 随机源
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.num_keypoints = num_keypoints

    def __call__(self, image, target):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(target['boxes'], rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (target['boxes'][:, :2] + target['boxes'][:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                # current_boxes = boxes[mask, :].copy()
                current_boxes = _copy_box(target['boxes'][mask, :])

                if 'keypoints' in target:
                    assert self.num_keypoints > 0, 'cfg.NUM_KEYPOINTS = %s 设置错误' % self.num_keypoints
                    target['keypoints'] = target['keypoints'][mask, :]
                    target['kps_mask'] = target['kps_mask'][mask, :]
                    # ltrb
                    rect_lt_x5 = np.tile(rect[:2], self.num_keypoints).astype(np.float32)
                    target['keypoints'] -= rect_lt_x5
                    img_wh_kp_ts = torch.tensor(current_image.shape[:2][::-1]).repeat(self.num_keypoints)
                    fverify_keypoints_2d(target, img_wh_kp_ts=img_wh_kp_ts)

                # take only matching gt labels
                current_labels = target['labels'][mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                target['labels'] = current_labels
                target['boxes'] = current_boxes

                return current_image, target


class FRandomMirror(object):
    '''随机水平镜像'''

    def __init__(self, data_name=None) -> None:
        super().__init__()
        self.name_data = data_name
        self.rules_flip = {
            'face98': {
                0: 32, 1: 31, 2: 30, 3: 29, 4: 28, 5: 27, 6: 26, 7: 25, 8: 24, 9: 23, 10: 22, 11: 21, 12: 20, 13: 19,
                14: 18,
                15: 17,
                16: 16, 17: 15, 18: 14, 19: 13, 20: 12, 21: 11, 22: 10, 23: 9, 24: 8, 25: 7, 26: 6, 27: 5, 28: 4, 29: 3,
                30: 2,
                31: 1, 32: 0,
                33: 46, 34: 45, 35: 44, 36: 43, 37: 42, 38: 50, 39: 49, 40: 48, 41: 47,
                46: 33, 45: 34, 44: 35, 43: 36, 42: 37, 50: 38, 49: 39, 48: 40, 47: 41,
                60: 72, 61: 71, 62: 70, 63: 69, 64: 68, 65: 75, 66: 74, 67: 73,
                72: 60, 71: 61, 70: 62, 69: 63, 68: 64, 75: 65, 74: 66, 73: 67,
                96: 97, 97: 96,
                51: 51, 52: 52, 53: 53, 54: 54,
                55: 59, 56: 58, 57: 57, 58: 56, 59: 55,
                76: 82, 77: 81, 78: 80, 79: 79, 80: 78, 81: 77, 82: 76,
                87: 83, 86: 84, 85: 85, 84: 86, 83: 87,
                88: 92, 89: 91, 90: 90, 91: 89, 92: 88,
                95: 93, 94: 94, 93: 95
            },
            'face5': {0: 1, -1: -1, 2: 2, 3: 4, -1: -1},  # 水平翻转 这个要和数据遍历对起
            'widerface': {0: 1, -1: -1, 2: 2, 3: 4, -1: -1},
        }

    def __call__(self, image, target):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            # boxes = boxes.copy()
            boxes = _copy_box(target['boxes'])
            # boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes

            if 'keypoints' in target:
                # if self.data_name is None:
                #     raise Exception('RandomMirror keypoints 不支持 需传入 data_name')
                assert self.name_data is not None, 'RandomMirror keypoints 不支持 需传入 name_data cfg.NAME_DATA '
                # 批量修改x值 y不变
                target['keypoints'][:, ::2] = width - target['keypoints'][:, ::2]

                ''' 这个需要订制一个翻转规则  '''
                if self.name_data not in self.rules_flip:
                    raise Exception('self.name_data 不存在 %s' % self.name_data)
                rule_flip = self.rules_flip[self.name_data]

                # 交换数组位置
                n = target['keypoints'].shape[0]
                kks = target['keypoints'].view(n, -1, 2)
                mms = target['kps_mask'].view(n, -1, 2)

                for i, _ in enumerate(kks):
                    # 改变 kk可以改变 target['keypoints'] 共享内存
                    kk = kks[i]
                    mm = mms[i]
                    for k, v in rule_flip.items():
                        if k == -1:
                            continue
                        _t = kk[k].clone()
                        kk[k] = kk[v]
                        kk[v] = _t
                        _t = mm[k].clone()
                        mm[k] = mm[v]
                        mm[v] = _t
                # 共享内存直接调整 无需这个
                # target['keypoints'] = kks.reshape(n, -1)
                # target['kps_mask'] = mms.reshape(n, -1)

            # target['keypoints'].view(-1,2).flip(0)

        return image, target


class FResize(object):
    def __init__(self, size_wh=None, is_multi_scale=False, multi_scale_val=(800, 1333)):
        '''
        如果 不是多尺寸 则按size_wh
        '''
        self.size_wh = size_wh
        self.is_multi_scale = is_multi_scale
        self.multi_scale_val = multi_scale_val

    def __call__(self, img_np, target):
        if self.is_multi_scale:
            min_side, max_side = self.multi_scale_val
            h, w, _ = img_np.shape
            smallest_side = min(w, h)
            largest_side = max(w, h)
            scale = min_side / smallest_side
            if largest_side * scale > max_side:
                scale = max_side / largest_side
            nw, nh = int(scale * w), int(scale * h)
            image_resized = cv2.resize(img_np, (nw, nh))

            if target is not None:
                target['toone'] = image_resized.shape[:2][::-1]
            pad_w = 32 - nw % 32
            pad_h = 32 - nh % 32
            # 右下加上pad 不影响targets
            image_res = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
            image_res[:nh, :nw, :] = image_resized

            scale_w = scale
            scale_h = scale
        else:
            assert self.size_wh is not None
            size_wh_z = self.size_wh
            w_ratio, h_ratio = np.array(img_np.shape[:2][::-1]) / np.array(size_wh_z)

            image_res = cv2.resize(img_np, (size_wh_z[0], size_wh_z[1]))

            scale_w = 1 / w_ratio
            scale_h = 1 / h_ratio

        if target is not None:
            if 'boxes' in target:
                target['boxes'][:, [0, 2]] = target['boxes'][:, [0, 2]] * scale_w
                target['boxes'][:, [1, 3]] = target['boxes'][:, [1, 3]] * scale_h
            if 'keypoints' in target:
                target['keypoints'][:, ::2] *= scale_w
                target['keypoints'][:, 1::2] *= scale_h

            # f_show_kp_np4plt(image_res, target['boxes'],
            #                  kps_xy_input=target['keypoints'],
            #                  mask_kps=target['kps_mask'],
            #                  is_recover_size=False)
            # f_show_od_pil4plt(image_res, target['boxes'], is_recover_size=False)
        return image_res, target


# --------------- 后处理 ------------
class Normalize(object):
    # def __init__(self, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    def __init__(self, mean=(128.0, 128.0, 128.0), std=(256.0, 256.0, 256.0)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, target):
        ''' -0.5 ~ 0.5 '''
        # bgr
        # f_show_pic_np4plt(image)

        # (928, 1216, 3)
        image = image.astype(np.float32).clip(0.0, 255.)
        # image /= 255.
        image -= self.mean.reshape(1, 1, -1)
        image /= self.std

        if image.max() > 0.5 or image.min() < -0.5:
            raise Exception('图片有问题 mean = %s ,std = %s, max=%s,min=%s'
                            % (str(self.mean), str(self.std), image.max(), image.min()))

        # f_show_pic_np4plt(image)
        # image = f_recover_normalization4np(image)
        # f_show_pic_np4plt(image)

        return image, target


class ConvertColor(object):
    '''bgr -> hsv'''

    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        # plt.imshow(image)
        # plt.show()
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError
        return image, target


class img_np_bgr2ts_rgb(object):
    def __init__(self, is_box_oned=False) -> None:
        super().__init__()
        self.is_box_oned = is_box_oned

    def __call__(self, cvimage, target):
        if target and self.is_box_oned:
            # np整体复制 wh
            whwh = np.tile(cvimage.shape[:2][::-1], 2)
            target['boxes'][:, :] = target['boxes'][:, :] / whwh
            if 'keypoints' in target:
                target['keypoints'][:, ::2] /= whwh[0]
                target['keypoints'][:, 1::2] /= whwh[1]

            # plt.imshow(cvimage)
            # plt.show()
        # (h,w,c -> c,h,w) = bgr
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), target


# ------------------------------ 无用 --------------------------------
class Target2one(object):
    '''原图转归一化尺寸  ToAbsoluteCoords 相反'''

    def __call__(self, image, target):
        height, width, channels = image.shape
        if target is not None:
            if 'boxes' in target:
                target['boxes'][:, 0] /= width
                target['boxes'][:, 2] /= width
                target['boxes'][:, 1] /= height
                target['boxes'][:, 3] /= height

            if 'keypoints' in target:
                target['keypoints'][:, ::2] /= width
                target['keypoints'][:, 1::2] /= height

        return image, target


class ToAbsoluteCoords(object):
    ''' boxes 恢复原图尺寸  归一化尺寸转原图  ToAbsoluteCoords 相反'''

    def __call__(self, image, target):
        '''
        归一化 -> 绝对坐标
        :param image:
        :param boxes:
        :param labels:
        :return:
        '''
        height, width, channels = image.shape
        if target is not None:
            if 'boxes' in target:
                # 这里可以直接改
                target['boxes'][:, 0] *= width
                target['boxes'][:, 2] *= width
                target['boxes'][:, 1] *= height
                target['boxes'][:, 3] *= height
            if 'keypoints' in target:
                target['keypoints'][:, ::2] *= width
                target['keypoints'][:, 1::2] *= height

        return image, target


class FPResize(object):
    def __init__(self, size, is_size_scope=False):
        '''
        size 可以是尺寸和范围 要求wh
        '''
        self.size = size
        self.is_size_scope = is_size_scope

    def __call__(self, img_np, target):
        if self.is_size_scope:
            min_side, max_side = self.size
            h, w, _ = img_np.shape
            smallest_side = min(w, h)
            largest_side = max(w, h)
            scale = min_side / smallest_side
            if largest_side * scale > max_side:
                scale = max_side / largest_side
            nw, nh = int(scale * w), int(scale * h)
            image_res = cv2.resize(img_np, (nw, nh))

            if target is not None:
                target['toone'] = image_res.shape[:2][::-1]

            scale_w = scale
            scale_h = scale
        else:
            w_ratio, h_ratio = np.array(img_np.shape[:2][::-1]) / np.array(self.size)

            image_res = cv2.resize(img_np, (self.size[0], self.size[1]))

            scale_w = 1 / w_ratio
            scale_h = 1 / h_ratio

        if target is not None:
            if 'boxes' in target:
                target['boxes'][:, [0, 2]] = target['boxes'][:, [0, 2]] * scale_w
                target['boxes'][:, [1, 3]] = target['boxes'][:, [1, 3]] * scale_h
            if 'keypoints' in target:
                target['keypoints'][:, ::2] *= scale_w
                target['keypoints'][:, 1::2] *= scale_h

        return image_res, target


class Lambda(object):
    """Applies a lambda as a transform. 这个用于复写方法?"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, target):
        return self.lambd(img, target)


class PhotometricDistort(object):
    '''图片增强'''

    def __init__(self):
        self.pd = [
            RandomContrast(),  # 随机透明度
            ConvertColor(transform='HSV'),  # bgr -> hsv
            RandomSaturation(),  # 随机色彩'
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),  # hsv -> bgr
            RandomContrast()  # 随机透明度
        ]
        self.rand_brightness = RandomBrightness()  # 随机亮度增强
        # self.rand_light_noise = RandomLightingNoise()  # 颜色杂音

    def __call__(self, image, target):
        im = image.copy()
        im, target = self.rand_brightness(im, target)
        if random.randint(2):  # 先转换还是后转换
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, target = distort(im, target)
        return im, target
    # return self.rand_light_noise(im, boxes, labels)


class SwapChannels(object):
    """
    随机 RGB 打乱
    Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps=(2, 1, 0)):
        self.swaps = swaps

    def __call__(self, image, target):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image, target


class KeyPointCropOne:
    '''
    支持 KeyPoint 随机剪图 + 归一化 图片没有归一化
    用这个 ToTensor is_box_oned=False
    '''

    def __init__(self, size_f, keep_scale_wh=False):
        self.keep_scale_wh = keep_scale_wh  # 这个标签不支持不能用
        self.size_f = size_f
        self.flip_landmarks_dict = {
            0: 32, 1: 31, 2: 30, 3: 29, 4: 28, 5: 27, 6: 26, 7: 25, 8: 24, 9: 23, 10: 22, 11: 21, 12: 20, 13: 19,
            14: 18,
            15: 17,
            16: 16, 17: 15, 18: 14, 19: 13, 20: 12, 21: 11, 22: 10, 23: 9, 24: 8, 25: 7, 26: 6, 27: 5, 28: 4, 29: 3,
            30: 2,
            31: 1, 32: 0,
            33: 46, 34: 45, 35: 44, 36: 43, 37: 42, 38: 50, 39: 49, 40: 48, 41: 47,
            46: 33, 45: 34, 44: 35, 43: 36, 42: 37, 50: 38, 49: 39, 48: 40, 47: 41,
            60: 72, 61: 71, 62: 70, 63: 69, 64: 68, 65: 75, 66: 74, 67: 73,
            72: 60, 71: 61, 70: 62, 69: 63, 68: 64, 75: 65, 74: 66, 73: 67,
            96: 97, 97: 96,
            51: 51, 52: 52, 53: 53, 54: 54,  # 这里不用换 79 90 94 等
            55: 59, 56: 58, 57: 57, 58: 56, 59: 55,
            76: 82, 77: 81, 78: 80, 79: 79, 80: 78, 81: 77, 82: 76,
            87: 83, 86: 84, 85: 85, 84: 86, 83: 87,
            88: 92, 89: 91, 90: 90, 91: 89, 92: 88,
            95: 93, 94: 94, 93: 95
        }

    def letterbox(self, img_, img_size=256, mean_rgb=(128, 128, 128)):
        shape_ = img_.shape[:2]  # shape = [height, width]
        ratio = float(img_size) / max(shape_)  # ratio  = old / new
        new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
        dw_ = (img_size - new_shape_[0]) / 2  # width padding
        dh_ = (img_size - new_shape_[1]) / 2  # height padding
        top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
        left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
        # resize img
        img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

        img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT,
                                   value=mean_rgb)  # padded square

        return img_a

    def __call__(self, image, target, site_xy, vis=False):
        '''

        :param image:
        :param target:
        :param site_xy: 旋转锚点
        :param vis:
        :return:
        '''
        cx, cy = site_xy
        pts = target['keypoints']
        angle = random.randint(-36, 36)

        (h, w) = image.shape[:2]
        h = h
        w = w
        # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算新图像的bounding
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += int(0.5 * nW) - cx
        M[1, 2] += int(0.5 * nH) - cy

        resize_model = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

        img_rot = cv2.warpAffine(image, M, (nW, nH), flags=resize_model[random.randint(0, 4)])
        # flags : INTER_LINEAR INTER_CUBIC INTER_NEAREST
        # borderMode : BORDER_REFLECT BORDER_TRANSPARENT BORDER_REPLICATE CV_BORDER_WRAP BORDER_CONSTANT

        pts_r = []
        for pt in pts:
            x = float(pt[0])
            y = float(pt[1])

            x_r = (x * M[0][0] + y * M[0][1] + M[0][2])
            y_r = (x * M[1][0] + y * M[1][1] + M[1][2])

            pts_r.append([x_r, y_r])

        x = [pt[0] for pt in pts_r]
        y = [pt[1] for pt in pts_r]

        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)

        translation_pixels = 60

        scaling = 0.3
        x1 += random.randint(-int(max((x2 - x1) * scaling, translation_pixels)), int((x2 - x1) * 0.25))
        y1 += random.randint(-int(max((y2 - y1) * scaling, translation_pixels)), int((y2 - y1) * 0.25))
        x2 += random.randint(-int((x2 - x1) * 0.15), int(max((x2 - x1) * scaling, translation_pixels)))
        y2 += random.randint(-int((y2 - y1) * 0.15), int(max((y2 - y1) * scaling, translation_pixels)))

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(x2, img_rot.shape[1] - 1))
        y2 = int(min(y2, img_rot.shape[0] - 1))

        crop_rot = img_rot[y1:y2, x1:x2, :]

        crop_pts = []
        width_crop = float(x2 - x1)
        height_crop = float(y2 - y1)
        # 归一化
        for pt in pts_r:
            x = pt[0]
            y = pt[1]
            crop_pts.append([float(x - x1) / width_crop, float(y - y1) / height_crop])

        # 随机镜像 这个存在标签左右眼切换问题 98点可以支持 其它需定制
        if random.random() >= 0.5:
            # print('--------->>> flip')
            crop_rot = cv2.flip(crop_rot, 1)
            crop_pts_flip = []
            for i in range(len(crop_pts)):
                # print( crop_rot.shape[1],crop_pts[flip_landmarks_dict[i]][0])
                x = 1. - crop_pts[self.flip_landmarks_dict[i]][0]
                y = crop_pts[self.flip_landmarks_dict[i]][1]
                # print(i,x,y)
                crop_pts_flip.append([x, y])
            crop_pts = crop_pts_flip

        if vis:
            for pt in crop_pts:
                x = int(pt[0] * width_crop)
                y = int(pt[1] * height_crop)

                cv2.circle(crop_rot, (int(x), int(y)), 2, (255, 0, 255), -1)
            # cv2.imshow('img', crop_rot)
            from matplotlib import pyplot as plt
            plt.imshow(crop_rot)
            plt.show()

        if self.keep_scale_wh:
            # 这个标签不支持不能用
            # crop_rot = self.letterbox(crop_rot, img_size=self.size_f[0], mean_rgb=(128, 128, 128))
            raise Exception('self.keep_scale_wh = %s 不支持 ' % self.keep_scale_wh)
        else:
            crop_rot = cv2.resize(crop_rot, self.size_f, interpolation=resize_model[random.randint(0, 4)])

        if vis:
            for pt in crop_pts:
                x = int(pt[0] * self.size_f[0])
                y = int(pt[1] * self.size_f[1])

                cv2.circle(crop_rot, (int(x), int(y)), 2, (255, 0, 255), -1)
            # cv2.imshow('img', crop_rot)
            from matplotlib import pyplot as plt
            plt.imshow(crop_rot)
            plt.show()
        ''' 图片没有归一化 '''
        return image, target


def f_recover_normalization4ts(img_ts, mean_bgr=(128.0, 128.0, 128.0), std_bgr=(256.0, 256.0, 256.0)):
    '''

    :param img_ts: c,h,w
    :return:
    '''
    device = img_ts.device
    # torch.Size([3, 928, 1536]) -> torch.Size([928, 1536, 3])
    # img_ts_show = img_ts.permute(1, 2, 0) # rgb -> bgr
    mean_bgr = torch.tensor(mean_bgr, device=device)[:, None, None]  # 3,1,1
    std_bgr = torch.tensor(std_bgr, device=device).unsqueeze(-1).unsqueeze(-1)
    img_ts = img_ts * std_bgr + mean_bgr
    # img_ts_show = img_ts_show.permute(2, 0, 1)# bgr -> rgb
    return img_ts


def f_recover_normalization4np(img_np_bgr, mean_bgr=(128.0, 128.0, 128.0), std_bgr=(256.0, 256.0, 256.0)):
    '''

    :param img_np_bgr: c,h,w
    :return:
    '''
    mean_bgr = np.array(mean_bgr)
    std_bgr = np.array(std_bgr)
    img_np_bgr = img_np_bgr * std_bgr + mean_bgr
    return img_np_bgr


def random_rotation4ts(img_np, boxes_ltrb_ts, kps_3d_ts, degree=10):
    '''
    kps_ts: torch.Size([1, 5, 3])
    '''
    angle = random.uniform(-degree, degree)
    h, w, c = img_np.shape
    cx, cy = w / 2.0, h / 2.0

    ''' 图片处理 '''
    # mat rotate 1 center 2 angle 3 缩放系数
    matRotate = cv2.getRotationMatrix2D((cy, cx), angle, 1.0)
    img_np = cv2.warpAffine(img_np, matRotate, (h, w))

    # debug
    # print(angle)
    # plt.imshow(img_np)
    # plt.show()

    ''' box处理 '''
    a = -angle / 180.0 * math.pi
    # boxes = torch.from_numpy(boxes)
    new_boxes_ltrb_ts = torch.zeros_like(boxes_ltrb_ts)
    new_boxes_ltrb_ts[:, 0] = boxes_ltrb_ts[:, 1]
    new_boxes_ltrb_ts[:, 1] = boxes_ltrb_ts[:, 0]
    new_boxes_ltrb_ts[:, 2] = boxes_ltrb_ts[:, 3]
    new_boxes_ltrb_ts[:, 3] = boxes_ltrb_ts[:, 2]
    for i in range(boxes_ltrb_ts.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes_ltrb_ts[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
        tp = torch.zeros_like(z)
        tp[:, 1] = (z[:, 1] - cx) * math.cos(a) - (z[:, 0] - cy) * math.sin(a) + cx
        tp[:, 0] = (z[:, 1] - cx) * math.sin(a) + (z[:, 0] - cy) * math.cos(a) + cy
        ymax, xmax = torch.max(tp, dim=0)[0]
        ymin, xmin = torch.min(tp, dim=0)[0]
        new_boxes_ltrb_ts[i] = torch.stack([ymin, xmin, ymax, xmax])
    new_boxes_ltrb_ts[:, 1::2].clamp_(min=0, max=h - 1)
    new_boxes_ltrb_ts[:, 0::2].clamp_(min=0, max=w - 1)
    boxes_ltrb_ts[:, 0] = new_boxes_ltrb_ts[:, 1]
    boxes_ltrb_ts[:, 1] = new_boxes_ltrb_ts[:, 0]
    boxes_ltrb_ts[:, 2] = new_boxes_ltrb_ts[:, 3]
    boxes_ltrb_ts[:, 3] = new_boxes_ltrb_ts[:, 2]
    # boxes_ts = boxes_ts.numpy()

    ngt, nkey, c = kps_3d_ts.shape
    for i in range(ngt):
        for j in range(nkey):
            x = kps_3d_ts[i][j][0]
            y = kps_3d_ts[i][j][1]
            p = np.array([x, y, 1])
            p = matRotate.dot(p)
            kps_3d_ts[i][j][0] = p[0]
            kps_3d_ts[i][j][1] = p[1]

    return img_np, boxes_ltrb_ts, kps_3d_ts


def random_rotation4np(img_np, boxes_ltrb_np, kps_3d_np, degree=10):
    angle = random.uniform(-degree, degree)
    h, w, c = img_np.shape
    cx, cy = w / 2.0, h / 2.0

    ''' 图片处理 '''
    # mat rotate 1 center 2 angle 3 缩放系数
    matRotate = cv2.getRotationMatrix2D((cy, cx), angle, 1.0)
    img_np = cv2.warpAffine(img_np, matRotate, (h, w))

    # debug
    # print(angle)
    # plt.imshow(img_np)
    # plt.show()

    ''' box处理 '''
    a = -angle / 180.0 * math.pi
    # boxes = torch.from_numpy(boxes)
    new_boxes_ltrb_np = np.zeros_like(boxes_ltrb_np)
    new_boxes_ltrb_np[:, 0] = boxes_ltrb_np[:, 1]
    new_boxes_ltrb_np[:, 1] = boxes_ltrb_np[:, 0]
    new_boxes_ltrb_np[:, 2] = boxes_ltrb_np[:, 3]
    new_boxes_ltrb_np[:, 3] = boxes_ltrb_np[:, 2]
    for i in range(boxes_ltrb_np.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes_ltrb_np[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = np.array([[y0, x0], [y1, x1], [y2, x2], [y3, x3]], dtype=np.float32)
        tp = np.zeros_like(z)
        tp[:, 1] = (z[:, 1] - cx) * math.cos(a) - (z[:, 0] - cy) * math.sin(a) + cx
        tp[:, 0] = (z[:, 1] - cx) * math.sin(a) + (z[:, 0] - cy) * math.cos(a) + cy
        ymax, xmax = np.max(tp, axis=0)
        ymin, xmin = np.min(tp, axis=0)
        new_boxes_ltrb_np[i] = np.stack([ymin, xmin, ymax, xmax])

    new_boxes_ltrb_np[:, 1::2] = np.clip(new_boxes_ltrb_np[:, 0::2], a_min=0, a_max=h - 1)
    new_boxes_ltrb_np[:, 1::2] = np.clip(new_boxes_ltrb_np[:, 0::2], a_min=0, a_max=w - 1)
    boxes_ltrb_np[:, 0] = new_boxes_ltrb_np[:, 1]
    boxes_ltrb_np[:, 1] = new_boxes_ltrb_np[:, 0]
    boxes_ltrb_np[:, 2] = new_boxes_ltrb_np[:, 3]
    boxes_ltrb_np[:, 3] = new_boxes_ltrb_np[:, 2]

    ngt, nkey, c = kps_3d_np.shape
    for i in range(ngt):
        for j in range(nkey):
            x = kps_3d_np[i][j][0]
            y = kps_3d_np[i][j][1]
            p = np.array([x, y, 1])
            p = matRotate.dot(p)
            kps_3d_np[i][j][0] = p[0]
            kps_3d_np[i][j][1] = p[1]

    return img_np, boxes_ltrb_np, kps_3d_np


if __name__ == '__main__':
    from f_tools.datas.f_coco.coco_dataset import CustomCocoDataset, fverify_keypoints_2d

    mode = 'keypoints'  # bbox segm keypoints caption


    # mode = 'bbox'  # bbox segm keypoints caption

    class cfg:
        pass


    cfg.IMAGE_SIZE = (448, 448)
    cfg.PIC_MEAN = (128, 128, 128.)
    cfg.PIC_STD = (256, 256, 256)
    cfg.USE_BASE4NP = False  # 基础处理
    cfg.IS_VISUAL_PRETREATMENT = False  # 用于dataset提取时测试
    cfg.NUM_KEYPOINTS = 5
    cfg.MODE_VIS = 'keypoints'
    cfg.NAME_DATA = 'face5'
    cfg.tcfg_pic_handler = ''
    cfg.IS_MULTI_SCALE_V2 = True
    cfg.MULTI_SCALE_VAL_V2 = [900, 2000]

    P_SIZE = [416, 416]
    IS_SIZE_SCOPE = False

    transform = None
    # transform = FExpand(cfg.PIC_MEAN, cfg.NUM_KEYPOINTS)
    # transform = SwapChannels()
    # transform = FRandomRotate(90)
    # transform = FRandomNoise()
    # transform = Ffilter_mean()
    # transform = Fgamma_transform()
    # transform = FRandomSampleCrop(cfg.NUM_KEYPOINTS)
    # transform = FResize([500, 100])
    # transform = FResize(is_multi_scale=True, multi_scale_val=[800, 1333])
    # transform = FPResize([300, 200])
    # transform = FPResize([100, 300], is_size_scope=True)
    # transform = RandomMirror()
    # transform = img_np_bgr2ts_rgb()

    # transform = cre_transform_np4train(cfg)['train']
    # transform = cre_transform_np4base(cfg)['train']
    # transform = cre_transform_balance_data(cfg)['train']
    # transform = Compose([
    #     # ConvertFromInts(),  # image int8转换成float [0,256)
    #     # ToAbsoluteCoords(), # 恢复真实尺寸
    #     # PhotometricDistort(),  # 图片处理集合
    #     FExpand(cfg.PIC_MEAN,cfg.NUM_KEYPOINTS),  # 放大缩小图片
    #     # FRandomSampleCrop(cfg.NUM_KEYPOINTS),  # 随机剪切定位
    #     # FRandomMirror('face5'),
    #     # ToPercentCoords(),  # boxes 按原图归一化 最后ToTensor 统一归一
    #     # Resize(cfg.IMAGE_SIZE),  # 定义模型输入尺寸 处理img和boxes
    #     Normalize(cfg.PIC_MEAN, cfg.PIC_STD),  # 正则化图片
    #     # ConvertColor(current='BGR', transform='RGB'),
    #     # ToTensor(is_box_oned=True),  # img 及 boxes(可选,前面已归一)归一  转tensor
    # ], cfg)

    # path_img, dataset = load_dataset_coco(mode, transform=transform)

    # path_root = r'M:\AI\datas\face_98'  # 自已的数据集
    # file_json = os.path.join(path_root, 'annotations', 'keypoints_test_2500_2118.json')
    # path_img = os.path.join(path_root, 'images_test_2118')

    # file_json = os.path.join(path_root, 'annotations', 'keypoints_train_7500_5316.json')
    # path_img = os.path.join(path_root, 'images_train_5316')

    # path_root = r'M:\AI\datas\face_5'
    path_root = r'/AI/datas/face_5'
    file_json = os.path.join(path_root, 'annotations', 'keypoints_train_10000_10000.json')
    path_img = os.path.join(path_root, 'images_13466')

    # path_root = r'/AI/datas/VOC2007'
    # file_json = os.path.join(path_root, 'coco/annotations', 'instances_type3_train_1066.json')
    # path_img = os.path.join(path_root, 'train/JPEGImages')

    dataset = CustomCocoDataset(
        file_json=file_json,
        path_img=path_img,
        mode=mode,
        transform=transform,
        cfg=cfg,
    )
    from f_tools.pic.f_show import f_show_od_np4plt_v1, fshow_kp_ts4plt, f_show_kp_np4plt, f_show_kp_np4plt_new

    for data in dataset:
        img_np_or_tensor, target = data

        if mode == 'keypoints':
            if isinstance(img_np_or_tensor, np.ndarray):
                f_show_kp_np4plt(img_np_or_tensor, target['boxes'],
                                 kps_xy_input=target['keypoints'],
                                 mask_kps=torch.ones_like(target['keypoints'], dtype=torch.bool),
                                 )

                kps_3d_ts = target['keypoints']
                b, c = kps_3d_ts.shape
                kps_3d_ts = kps_3d_ts.reshape(b, 5, 2)
                a1 = torch.ones(b, 5, 1)
                # torch.Size([1, 5, 3])
                kps_3d_ts = torch.cat([kps_3d_ts, a1], -1)
                kps_3d_np = kps_3d_ts.numpy()
                # img_np_or_tensor, target['boxes'], kps_ts \
                #     = random_rotation4ts(img_np_or_tensor, target['boxes'], kps_ts, 90)

                img_np_or_tensor, target['boxes'], kps_3d_np \
                    = random_rotation4np(img_np_or_tensor, target['boxes'].numpy(), kps_3d_np, 90)

                f_show_kp_np4plt_new(img_np_or_tensor, target['boxes'],
                                     kps_xy_input=kps_3d_ts,
                                     )

            else:
                fshow_kp_ts4plt(img_np_or_tensor,
                                target['boxes'],
                                target['keypoints'],
                                mask_kps=target['kps_mask'],
                                is_recover_size=True
                                )  # 需要恢复box
        else:
            # img_np_or_tensor = f_recover_normalization4ts(img_np_or_tensor)
            f_show_od_np4plt_v1(img_np_or_tensor, target['boxes'],
                                # g_ltrb=targets[i]['boxes'].cpu() / size_wh_toone_ts_x2 * size_wh_file_x2,
                                # 这里有错? GT默认不归一化
                                # ids2classes=data_loader.dataset.ids_classes,
                                # labels=p_labels[mask],
                                # scores=p_scores[mask].tolist(),
                                is_recover_size=False
                                )

        # img_np_or_tensor, target['boxes'] = random_rotation_np(img_np_or_tensor, target['boxes'])
        # f_show_od_ts4plt_v3(img_np_or_tensor, target['boxes'], is_recover_size=False)

        # f_show_kp_np4plt(img_np_or_tensor, target['boxes'],
        #                  kps_xy_input=target['keypoints'],
        #                  mask_kps=target['kps_mask'],
        #                  # 测试集 GT默认不归一化,是input模型尺寸
        #                  # g_boxes_ltrb=targets[i]['boxes'].cpu() / _size * wh_x2,
        #                  # ids2classes=data_loader.dataset.ids_classes,
        #                  # labels=p_labels[mask],
        #                  # scores=p_scores[mask].tolist(),
        #                  is_recover_size=False)

        # f_show_od_np4plt_v1(img_np_or_tensor, target['boxes'],
        #                     # g_ltrb=targets[i]['boxes'].cpu() / size_wh_toone_ts_x2 * size_wh_file_x2,
        #                     # 这里有错? GT默认不归一化
        #                     # ids2classes=data_loader.dataset.ids_classes,
        #                     # labels=p_labels[mask],
        #                     # scores=p_scores[mask].tolist(),
        #                     is_recover_size=False
        #                     )
