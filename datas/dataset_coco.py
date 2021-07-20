import os

import cv2
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

from tools.GLOBAL_LOG import flog


class CLS4collate_fn:
    # fdatas_l1
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch_datas):
        '''
        在这里重写collate_fn函数
        batch_datas: tuple[[tensor_img,dict_targets],...,[tensor_img,dict_targets]]
        '''
        # 训练才进这里
        if self.cfg.IS_MULTI_SCALE_V2:
            batch = len(batch_datas)
            imgs_list = []
            targets_list = []
            # 打开 tuple 数据
            for i, (img_ts, target) in enumerate(batch_datas):
                # flog.warning('fun4dataloader测试  %s %s %s ', target, len(target['boxes']), len(target['labels']))
                imgs_list.append(img_ts)
                targets_list.append(target)

            pad_imgs_list = []

            # 这里的最大一定能被32整除
            h_list = [int(s.shape[1]) for s in imgs_list]
            w_list = [int(s.shape[2]) for s in imgs_list]
            max_h = np.array(h_list).max()
            max_w = np.array(w_list).max()
            # self.cfg.tcfg_batch_size = [max_w, max_h] # 这样用多进程要报错
            for i in range(batch):
                img_ts = imgs_list[i]
                # 右下角添加 target 无需处理
                img_ts_pad = F.pad(img_ts, (0, int(max_w - img_ts.shape[2]), 0, int(max_h - img_ts.shape[1])), value=0.)
                pad_imgs_list.append(img_ts_pad)

                # debug 代码
                # fshow_pic_ts4plt(pad_img)  # 可视化 前面不要归一化
                # fshow_kp_ts4plt(pad_img,
                #                 targets_list[i]['boxes'],
                #                 targets_list[i]['keypoints'],
                #                 mask_kps=targets_list[i]['kps_mask'],
                #                 is_recover_size=False
                #                 )  # 可视化
                # f_show_od_ts4plt(img_ts_pad, targets_list[i]['boxes'], is_recover_size=False)
                # print('多尺度%s ' % str(pad_img.shape))

            imgs_ts_4d = torch.stack(pad_imgs_list)
        else:
            imgs_ts_3d = batch_datas[0][0]
            # 包装整个图片数据集 (batch,3,416,416) 转换到显卡
            imgs_ts_4d = torch.empty((len(batch_datas), *imgs_ts_3d.shape)).to(imgs_ts_3d)
            targets_list = []
            for i, (img, target) in enumerate(batch_datas):
                # flog.warning('fun4dataloader测试  %s %s %s ', target, len(target['boxes']), len(target['labels']))
                imgs_ts_4d[i] = img
                targets_list.append(target)
        return imgs_ts_4d, targets_list


class CustomCocoDataset(Dataset):
    '''
    v 20210718
    '''

    def __init__(self, cfg, file_json, path_img, mode, transform=None, is_debug=False,
                 s_ids_cats=None, nums_cat=None, is_ts_all=True,
                 image_size=None, mode_balance_data=None, name='train', ):
        '''

        :param cfg:
        :param file_json:
        :param path_img:
        :param mode:  bbox segm keypoints caption
        :param transform:
        :param is_debug:
        :param s_ids_cats:  指定类别数
        :param nums_cat:  限制类别的最大数量
        :param is_ts_all:  默认全部转TS
        :param image_size:
        :param mode_balance_data:  数据类别平衡方法
            'max': 4舍五入倍数整体复制  确保类别尽量一致
            'min': 取最少的类型数,多的随机选
            None: 不处理
        :param name: dataset 标注
        '''
        assert cfg is not None, 'CustomCocoDataset  cfg 不能为空'
        self.image_size = image_size

        self.file_json = file_json
        self.transform = transform
        self.mode = mode
        self.coco_obj = COCO(file_json)

        self.name = name
        self.is_ts_all = is_ts_all

        # f_look_coco_type(self.coco_obj, ids_cats_ustom=None)
        print('创建dataset-----', name)

        if mode_balance_data is not None:
            cats = self.coco_obj.getCatIds()
            num_class = len(cats)
            _t_ids = []
            _t_len = np.zeros(num_class)
            for i, cat in enumerate(cats):
                ids = self.coco_obj.getImgIds(catIds=cat)
                _t_ids.append(ids)
                _t_len[i] = len(ids)

            self.ids_img = []

            if mode_balance_data == 'max':
                num_repeats = np.around((_t_len.max() / _t_len)).astype(np.int)
                for i in range(num_class):
                    self.ids_img.extend(np.tile(np.array(_t_ids[i]), num_repeats[i]).tolist())
            elif mode_balance_data == 'min':
                len_min = _t_len.min().astype(np.int)
                # flog.debug('_t_len = %s', _t_len)
                for i in range(num_class):
                    self.ids_img.extend(np.random.choice(_t_ids[i], len_min).tolist())

        else:
            if s_ids_cats is not None:
                flog.warning('指定coco类型 %s', self.coco_obj.loadCats(s_ids_cats))
                self.s_ids_cats = s_ids_cats
                ids_img = []

                # 限制每类的最大个数
                if nums_cat is None:
                    for idc in zip(s_ids_cats):
                        # 类型对应哪些文件 可能是一张图片多个类型
                        ids_ = self.coco_obj.getImgIds(catIds=idc)
                        ids_img += ids_
                else:
                    # 限制每类的最大个数
                    for idc, num_cat in zip(s_ids_cats, nums_cat):
                        # 类型对应哪些文件 可能是一张图片多个类型
                        ids_ = self.coco_obj.getImgIds(catIds=idc)[:num_cat]
                        # ids_ = self.coco.getImgIds(catIds=idc)[:1000]
                        ids_img += ids_
                        # print(ids_)  # 这个只支持单个元素

                self.ids_img = list(set(ids_img))  # 去重
            else:
                self.ids_img = self.coco_obj.getImgIds()  # 所有图片的id 画图数量

        #  创建 coco 类别映射
        self._init_load_classes()  # 除了coco数据集,其它不管

        self.is_debug = is_debug
        self.cfg = cfg
        self.path_img = path_img
        if not os.path.exists(path_img):
            raise Exception('coco path_img 路径不存在', path_img)

    def __len__(self):
        return len(self.ids_img)

    def open_img_tar(self, id_img):
        img = self.load_image(id_img)

        # bboxs, labels,keypoints
        tars_ = self.load_anns(id_img, img_wh=img.shape[:2][::-1])
        if tars_ is None:  # 没有标注返回空
            return None

        # 动态构造target
        target = {}
        l_ = ['boxes', 'labels', 'keypoints', 'kps_mask']
        target['image_id'] = id_img
        target['size'] = np.array(img.shape[:2][::-1])  # (w,h)

        # 根据标注模式 及字段自动添加 target['boxes', 'labels', 'keypoints']
        for i, tar in enumerate(tars_):
            target[l_[i]] = tar
        return img, target

    def __getitem__(self, index):
        '''

        :param index:
        :return: tensor or np.array 根据 out: 默认ts or other is np
            img: h,w,c
            target:
            coco原装是 ltwh
            dict{
                image_id: int,
                bboxs: ts n4 原图 ltwh -> ltrb
                labels: ts n,
                keypoints: ts n,10
                size: wh
            }
        '''
        # 这里生成的是原图尺寸的 target 和img_np_uint8 (375, 500, 3)
        id_img = self.ids_img[index]
        res = self.open_img_tar(id_img)

        if res is None:
            print('这个图片没有标注信息 id为 %s ,继续下一个', id_img)
            return self.__getitem__(index + 1)

        img, target = res

        _text_base = '!!! 数据有问题 %s  %s %s %s '
        assert len(target['boxes']) == len(target['labels']), \
            _text_base % ('transform前', target, len(target['boxes']), len(target['labels']))

        if target['boxes'].shape[0] == 0:
            flog.warning('数据有问题 重新加载 %s', id_img)
            return self.__getitem__(index + 1)

        # 以上img 确定是 np格式(transform出来一般是ts); target 全部转np
        if self.transform is not None:
            img, target = self.transform(img, target)
        # 这里img输出 ts_3d
        if self.is_ts_all:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['size'] = torch.tensor(target['size'], dtype=torch.float)  # file尺寸

        assert len(target['boxes']) == len(target['labels']), \
            _text_base % ('transform后', target, len(target['boxes']), len(target['labels']))
        # 每个图片对应的target数量是不一致的 所以需要用target封装
        return img, target

    def load_image(self, id_img):
        '''

        :param id_img:
        :return:
        '''
        image_info = self.coco_obj.loadImgs(id_img)[0]
        file_img = os.path.join(self.path_img, image_info['file_name'])
        if not os.path.exists(file_img):
            raise Exception('file_img 加载图片路径错误', file_img)

        img = cv2.imread(file_img)
        return img

    def load_anns(self, id_img, img_wh):
        '''
        ltwh --> ltrb
        :param id_img:
        :return:
            bboxs: np(num_anns, 4)
            labels: np(num_anns)
        '''
        # annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        annotation_ids = self.coco_obj.getAnnIds(id_img)  # ann的id
        # anns is num_anns x 4, (x1, x2, y1, y2)
        bboxs_np = np.zeros((0, 4), dtype=np.float32)  # np创建 空数组 高级
        labels = []
        if len(annotation_ids) == 0:
            return None

        coco_anns = self.coco_obj.loadAnns(annotation_ids)
        for ann in coco_anns:
            x, y, box_w, box_h = ann['bbox']  # ltwh
            # 修正超范围的框  得 ltrb
            x1 = max(0, x)  # 修正lt最小为0 左上必须在图中
            y1 = max(0, y)
            x2 = min(img_wh[0] - 1, x1 + max(0, box_w - 1))  # 右下必须在图中
            y2 = min(img_wh[1] - 1, y1 + max(0, box_h - 1))
            ''' bbox校验 '''
            if ann['area'] > 0 and x2 > x1 and y2 >= y1:
                bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                # bbox[0, :4] = ann['bbox']
                # ann['bbox'] = [x1, y1, x2, y2]  # 这样写回有BUG 共享内存会修改
            else:
                flog.error('标记框有问题 %s 跳过', ann)
                continue

            # 全部通过后添加
            bboxs_np = np.append(bboxs_np, bbox, axis=0)
            labels.append(self.classes_coco2train[ann['category_id']])

        # bboxs = ltwh2ltrb(bboxs) # 前面 已转
        if bboxs_np.shape[0] == 0:
            flog.error('这图标注 不存在 %s', coco_anns)
            return None
            # raise Exception('这图标注 不存在 %s', coco_anns)

        # 这里转tensor
        if self.mode == 'bbox':
            return [bboxs_np, labels]
        elif self.mode == 'keypoints':
            pass

    def _init_load_classes(self):
        '''
        self.classes_ids :  {'Parade': 1}
        self.ids_classes :  {1: 'Parade'}
        self.ids_new_old {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        self.ids_old_new {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        :return:
        '''
        # [{'id': 1, 'name': 'aeroplane'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'bird'}, {'id': 4, 'name': 'boat'}, {'id': 5, 'name': 'bottle'}, {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'car'}, {'id': 8, 'name': 'cat'}, {'id': 9, 'name': 'chair'}, {'id': 10, 'name': 'cow'}, {'id': 11, 'name': 'diningtable'}, {'id': 12, 'name': 'dog'}, {'id': 13, 'name': 'horse'}, {'id': 14, 'name': 'motorbike'}, {'id': 15, 'name': 'person'}, {'id': 16, 'name': 'pottedplant'}, {'id': 17, 'name': 'sheep'}, {'id': 18, 'name': 'sofa'}, {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'tvmonitor'}]
        categories = self.coco_obj.loadCats(self.coco_obj.getCatIds())
        categories.sort(key=lambda x: x['id'])  # 按id升序 [{'id': 1, 'name': 'Parade'}]

        # coco ids is not from 1, and not continue ,make a new index from 0 to 79, continuely
        # 重建index 从1-80
        # classes_ids:   {names:      new_index}
        # coco_ids:  {new_index:  coco_index}
        # coco_ids_inverse: {coco_index: new_index}

        self.classes_ids, self.classes_train2coco, self.classes_coco2train = {}, {}, {}
        self.ids_classes = {}
        # 解决中间有断格的情况
        for i, c in enumerate(categories, start=1):  # 修正从1开始
            self.classes_train2coco[i] = c['id']  # 验证时用这个
            self.classes_coco2train[c['id']] = i
            self.classes_ids[c['name']] = i  # 这个是新索引 {'Parade': 0}
            self.ids_classes[i] = c['name']
        pass


if __name__ == '__main__':
    class cfg:
        pass


    path_root = 'M:/AI/datas/VOC2012'
    path_img = os.path.join(path_root, 'train/JPEGImages')
    file_json = os.path.join(path_root, 'coco/annotations/instances_train_17125.json')
    mode = 'bbox'  # bbox segm keypoints caption
    transform = None

    dataset = CustomCocoDataset(
        cfg=cfg,
        file_json=file_json,
        path_img=path_img,
        mode=mode,
        transform=transform,
        mode_balance_data=None,
    )

    print(dataset[0])
    print('len(dataset)', len(dataset))
