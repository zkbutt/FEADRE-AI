import math
import torch
import numpy as np


def empty_bboxes(bboxs):
    if isinstance(bboxs, np.ndarray):
        bboxs_copy = np.empty_like(bboxs, dtype=np.float)
    elif isinstance(bboxs, torch.Tensor):
        device = bboxs.device
        bboxs_copy = torch.empty_like(bboxs, device=device, dtype=torch.float)
    else:
        raise Exception('类型错误', type(bboxs))
    return bboxs_copy


def ltrb2xywh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)  # 复制矩阵
    # wh = rb - lt
    bboxs_copy[..., 2:] = bboxs[..., 2:] - bboxs[..., :2]
    # xy = lt + 0.5* cwh    or (lt+rb)/2
    bboxs_copy[..., :2] = bboxs[..., :2] + 0.5 * bboxs_copy[..., 2:]
    return bboxs_copy


def ltrb2ltwh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)
    # lt = lt
    bboxs_copy[..., :2] = bboxs[..., :2]
    # wh = rb -lt
    bboxs_copy[..., 2:] = bboxs[..., 2:] - bboxs[..., :2]
    return bboxs_copy


def ltwh2ltrb(bboxs):
    bboxs_copy = empty_bboxes(bboxs)
    # lt = lt
    bboxs_copy[..., :2] = bboxs[..., :2]
    # rb= lt +wh
    bboxs_copy[..., 2:] = bboxs[..., :2] + bboxs[..., 2:]
    return bboxs_copy


def xywh2ltrb(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    # if isinstance(bboxs_copy, np.ndarray):
    #     fdiv = np.true_divide
    #     v = 2
    # elif isinstance(bboxs_copy, torch.Tensor):
    #     fdiv = torch.true_divide
    #     v = torch.tensor(2, device=bboxs.device)
    # else:
    #     raise Exception('类型错误', type(bboxs))

    # lt = xy - wh/2
    bboxs_copy[..., :2] = bboxs[..., :2] - bboxs[..., 2:] * 0.5
    # rb = clt + wh
    bboxs_copy[..., 2:] = bboxs_copy[..., :2] + bboxs[..., 2:]
    return bboxs_copy


def xywh2ltwh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    if isinstance(bboxs_copy, np.ndarray):
        fdiv = np.true_divide
        v = 2
    elif isinstance(bboxs_copy, torch.Tensor):
        fdiv = torch.true_divide
        v = torch.tensor(2, device=bboxs.device)
    else:
        raise Exception('类型错误', type(bboxs))

    # lt = xy - wh/2
    bboxs_copy[..., :2] = bboxs[..., :2] - fdiv(bboxs[..., 2:], v)
    # wh=wh
    bboxs_copy[..., 2:] = bboxs[..., 2:]
    return bboxs_copy


def ltwh2xywh(bboxs):
    bboxs_copy = empty_bboxes(bboxs)

    # xy= lt+ wh*0.5
    bboxs_copy[..., :2] = bboxs[..., :2] + bboxs[..., 2:] * 0.5
    # wh=wh
    bboxs_copy[..., 2:] = bboxs[..., 2:]
    return bboxs_copy


def bbox_iou4np(bbox_a, bbox_b):
    '''
    求所有bboxs 与所有标定框 的交并比 ltrb
    返回一个数
    :param bbox_a: 多个预测框 (n,4)
    :param bbox_b: 多个标定框 (k,4)
    :return: <class 'tuple'>: (2002, 2) (n,k)
    '''
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    '''
    确定交叉部份的坐标  以下设 --- n = 3 , k = 4 ---
    广播 对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值 输出 (n,k,2)
    左上 右下
    '''
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # (3,1,2) (4,2)--->(3,4,2)
    # 选出前两个最小的 ymin，xmin 左上角的点 后面的通过广播
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    '''
    相交的面积 只有当右下的最小坐标  >>(xy全大于) 左上的最大坐标时才相交 用右下-左上得长宽
    '''
    # (2002,2,2)的每一第三维 降维运算(2002,2)  通过后面的是否相交的 降维运算 (2002,2)赛选
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # (2002,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2002)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    # (2,2) axis=1 2->1 行相乘 长*宽 --->降维运算(2)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    #  (2002,1) +(2) = (2002,2) 每个的二维面积
    _a = area_a[:, None] + area_b
    _area_all = (_a - area_i)  # (2002,2)-(2002,2)
    return area_i / _area_all  # (2002,2)


def bbox_iou_v3(boxes1, boxes2, mode='iou', is_aligned=False):
    '''
    overlaps
    :param boxes1: 支持 [n,4]^^[n,4] 或 [n,4]^^[m,4]
    :param boxes2:
    :param mode: 'iou' 'giou'
    :param is_aligned:
    :return:
    # 3,4
    boxes1 = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        # [32, 32, 38, 42],
    ])

    # 2,4
    boxes2 = torch.FloatTensor([
        [0, 0, 10, 20],
        [0, 10, 10, 19],
        # [10, 10, 20, 20],
    ])
    print(bbox_overlaps(boxes1=boxes1, boxes2=boxes2, mode='giou', is_aligned=True))
    结果 -------------------
    iou
    tensor([[0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000],
            [0.0000, 0.0000, 0.0000]])
    giou
    tensor([[0.5000, 0.0000, -0.5000],
            [-0.2500, -0.0500, 1.0000],
            [-0.8371, -0.8766, -0.8214]])
    '''

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (boxes1.size(-1) == 4 or boxes1.size(0) == 0)
    assert (boxes2.size(-1) == 4 or boxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert boxes1.shape[:-2] == boxes2.shape[:-2]
    eps = torch.finfo(torch.float16).eps
    batch_shape = boxes1.shape[:-2]

    rows = boxes1.size(-2)
    cols = boxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return boxes1.new(batch_shape + (rows,))
        else:
            return boxes1.new(batch_shape + (rows, cols))

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    if is_aligned:
        lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
            enclosed_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    else:
        lt = torch.max(boxes1[..., :, None, :2],
                       boxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(boxes1[..., :, None, 2:],
                       boxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(boxes1[..., :, None, :2],
                                    boxes2[..., None, :, :2])
            enclosed_rb = torch.max(boxes1[..., :, None, 2:],
                                    boxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def bbox_iou4aligned(box1_ltrb, box2_ltrb, is_giou=False, is_diou=False, is_ciou=False):
    '''
    这个是一一对应  box1  box2 支持2D和3D
    :param box1_ltrb:
    :param box2_ltrb:
    iou: 目标框不相交时全为0 不能反映如何相交的  --- 重叠面积
    giou: 当目标框完全包裹预测框的时候 退化为iou   --- 中心点距离
    diou: 提高收敛速度及全包裹优化
    ciou: --- 长宽比  CIOU有点不一致
    :return: n
    '''
    eps = torch.finfo(torch.float16).eps
    # torch.Size([2, 845, 2])
    max_lt = torch.max(box1_ltrb[..., :2], box2_ltrb[..., :2])  # left-top [N,M,2] 多维组合用法
    min_rb = torch.min(box1_ltrb[..., 2:], box2_ltrb[..., 2:])  # right-bottom [N,M,2]
    # 确保面积不能为负  否则以0计算
    inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    # inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    # inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # [N,M] 降维
    inter_area = torch.prod(inter_wh, dim=-1)  # 这个一定>=0

    # 并的面积 单项面积为负 则置0
    # area1 = (box1_ltrb[..., 2] - box1_ltrb[..., 0]) * (box1_ltrb[..., 3] - box1_ltrb[..., 1]).clamp(0)
    # area2 = (box2_ltrb[..., 2] - box2_ltrb[..., 0]) * (box2_ltrb[..., 3] - box2_ltrb[..., 1]).clamp(0)
    area1 = torch.prod(box1_ltrb[..., 2:] - box1_ltrb[..., :2], -1).clamp(0)
    area2 = torch.prod(box2_ltrb[..., 2:] - box2_ltrb[..., :2], -1).clamp(0)
    ''' 确保这个不为负 '''
    union_area = (area1 + area2 - inter_area).clamp(eps)  # A+B-交=并

    iou = inter_area / union_area  # 交一定小于并
    # flog.debug('iou %s', iou)

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(box1_ltrb[..., :2], box2_ltrb[..., :2])
        max_rb = torch.max(box1_ltrb[..., 2:], box2_ltrb[..., 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            ''' 确保这个不为负 '''
            max_area = (max_wh[..., 0] * max_wh[..., 1]).clamp(eps)
            giou = iou - (max_area - union_area) / max_area
            return giou

        ''' 确保这个不为负 '''
        c2 = (max_wh[..., 0] ** 2 + max_wh[..., 1] ** 2).clamp(eps)  # 最大矩形的矩离的平方
        box1_xywh = ltrb2xywh(box1_ltrb)
        box2_xywh = ltrb2xywh(box2_ltrb)
        xw2_xh2 = torch.pow(box1_xywh[..., :2] - box2_xywh[..., :2], 2)  # 中心点距离的平方
        d2 = xw2_xh2[..., 0] + xw2_xh2[..., 1]
        dxy = d2 / c2  # 中心比例距离
        if is_diou:
            diou = iou - dxy
            return diou

        if is_ciou:
            # [3, 1] / [3, 1]  => [3, 1]
            box1_atan_wh = torch.atan(box1_xywh[..., 2:3] / box1_xywh[..., 3:])
            box2_atan_wh = torch.atan(box2_xywh[..., 2:3] / box2_xywh[..., 3:])
            # torch.Size([3, 1])
            v = torch.pow(box1_atan_wh - box2_atan_wh, 2) * (4 / math.pi ** 2)
            v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            # v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            ciou = iou - (dxy + v * alpha)
            return ciou


def bbox_iou4unaligned(boxes1, boxes2, is_giou=False, is_diou=False, is_ciou=False):
    '''

    :param boxes1:torch.Size([m, 4]) ltrb
    :param boxes2:torch.Size([n, 4]) ltrb
    :param is_giou: 重合度++
    :param is_diou: +中心点
    :param is_ciou: +宽高
    :return:(m,n)
    '''
    # 交集面积
    max_lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2] 多维组合用法
    min_rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
    inter_wh = (min_rb - max_lt).clamp(min=0)  # [N,M,2]
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N,M] 降维

    # 并的面积
    area1 = bbox_area(boxes1)  # 降维 n
    area2 = bbox_area(boxes2)  # 降维 m
    union_area = area1[:, None] + area2 - inter_area + torch.finfo(torch.float16).eps  # 升维n m

    iou = inter_area / union_area

    if not (is_giou or is_diou or is_ciou):
        return iou
    else:
        # 最大矩形面积
        min_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        max_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        max_wh = max_rb - min_lt
        if is_giou:
            max_area = max_wh[:, :, 0] * max_wh[:, :, 1] + torch.finfo(torch.float16).eps  # 降维运算
            giou = iou - (max_area - union_area) / max_area
            return giou

        c2 = max_wh[:, :, 0] ** 2 + max_wh[:, :, 1] ** 2 + torch.finfo(torch.float16).eps  # 最大矩形的矩离的平方
        box1_xywh = ltrb2xywh(boxes1)
        box2_xywh = ltrb2xywh(boxes2)
        xw2_xh2 = torch.pow(box1_xywh[:, None, :2] - box2_xywh[:, :2], 2)  # 中心点距离的平方
        d2 = xw2_xh2[:, :, 0] + xw2_xh2[:, :, 1]
        dxy = d2 / c2  # 中心比例距离
        if is_diou:
            diou = iou - dxy
            return diou

        if is_ciou:
            box1_atan_wh = torch.atan(box1_xywh[:, 2:3] / box1_xywh[:, 3:])  # w/h
            box2_atan_wh = torch.atan(box2_xywh[:, 2:3] / box2_xywh[:, 3:])
            # torch.squeeze(ts)
            v = torch.pow(box1_atan_wh[:, None, :] - box2_atan_wh, 2) * (4 / math.pi ** 2)
            # v = torch.squeeze(v, -1)  # m,n,1 -> m,n 去掉最后一维
            v.squeeze_(-1)  # m,n,1 -> m,n 去掉最后一维
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            ciou = iou - (dxy + v * alpha)
            return ciou


def bbox_iou4fcos(poff_ltrb_exp, goff_ltrb):
    w_gt = goff_ltrb[:, :, 0] + goff_ltrb[:, :, 2]
    h_gt = goff_ltrb[:, :, 1] + goff_ltrb[:, :, 3]
    w_pred = poff_ltrb_exp[:, :, 0] + poff_ltrb_exp[:, :, 2]
    h_pred = poff_ltrb_exp[:, :, 1] + poff_ltrb_exp[:, :, 3]
    S_gt = w_gt * h_gt
    S_pred = w_pred * h_pred
    I_h = torch.min(goff_ltrb[:, :, 1], poff_ltrb_exp[:, :, 1]) + torch.min(goff_ltrb[:, :, 3], poff_ltrb_exp[:, :, 3])
    I_w = torch.min(goff_ltrb[:, :, 0], poff_ltrb_exp[:, :, 0]) + torch.min(goff_ltrb[:, :, 2], poff_ltrb_exp[:, :, 2])
    S_I = I_h * I_w
    U = S_gt + S_pred - S_I + 1e-20
    IoU = S_I / U

    return IoU


def bbox_area(boxes):
    ''' boxes 一定有个面积 '''
    return torch.abs(boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


if __name__ == '__main__':
    # 3,4
    boxes1 = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        # [32, 32, 38, 42],
    ])

    # 2,4
    boxes2 = torch.FloatTensor([
        [0, 0, 10, 20],
        [0, 10, 10, 19],
        [10, 10, 20, 20],
    ])
    print(bbox_iou_v3(boxes1=boxes1, boxes2=boxes2, mode='giou', is_aligned=False))
    print(bbox_iou4unaligned(boxes1=boxes1, boxes2=boxes2, is_giou=True))
