import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from tools.GLOBAL_LOG import flog


class FModelBase(nn.Module):
    def __init__(self, cfg, net, losser, preder):
        super(FModelBase, self).__init__()
        self.cfg = cfg
        self.net = net
        self.losser = losser
        self.preder = preder

    def forward(self, reses):
        imgs_ts_4d, targets = reses
        outs = self.net(imgs_ts_4d)

        ''' 多尺度图片归一化支持 '''
        device = imgs_ts_4d.device
        batch, c, h, w = imgs_ts_4d.shape
        toones_wh_ts_input = []
        if targets is not None and 'toone' in targets[0]:  # 多尺度归一
            for target in targets:
                toones_wh_ts_input.append(torch.tensor(target['toone'], device=device))
            toones_wh_ts_input = torch.stack(toones_wh_ts_input, 0)
        else:
            toones_wh_ts_input = torch.empty(batch, 2, device=device, dtype=torch.float)  # 实际为准
            toones_wh_ts_input[:, :] = torch.tensor(imgs_ts_4d.shape[2:4][::-1], device=device)

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            loss_total, log_dict = self.losser(outs, targets, imgs_ts_4d, toones_wh_ts_input)
            '''------验证loss 待扩展------'''

            return loss_total, log_dict
        else:
            with torch.no_grad():  # 这个没用
                # outs模型输出  input_image
                ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores = self.preder(outs, imgs_ts_4d, targets,
                                                                                       toones_wh_ts_input)
            return ids_batch, p_boxes_ltrb, p_keypoints, p_labels, p_scores


class AverageMeter(object):
    """ Computes ans stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FitExecutor():
    def __init__(self, cfg, model, fun_loss, save_weight_name, path_save_weight,
                 optimizer=None, dataloader_train=None, lr_scheduler=None,
                 end_epoch=None, is_mixture_fit=True, lr_val_base=1e-3,
                 device=torch.device('cpu'), is_writer=False,
                 num_save_interval=10, print_freq=10,
                 dataloader_val=None, val_freq=1,
                 ) -> None:
        '''

        :param model:
        :param fun_loss: 回调 loss 处理
        :param save_weight_name:
        :param path_save_weight:
        :param optimizer:
        :param dataloader_train:
        :param lr_scheduler:
        :param end_epoch:
        :param is_mixture_fit:
        :param lr_val_base:
        :param device:
        :param is_writer:
        :param num_save_interval:
        :param print_freq:
        :param dataloader_val:
        :param val_freq: 验证频率
        '''
        super().__init__()
        self.dataloader_train = dataloader_train
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.end_epoch = end_epoch
        self.is_mixture_fit = is_mixture_fit
        self.lr_val_base = lr_val_base
        self.device = device

        self.fun_loss = fun_loss
        self.cfg = cfg

        # 验证
        self.dataloader_val = dataloader_val
        self.val_freq = val_freq

        self.save_weight_name = save_weight_name
        self.path_save_weight = path_save_weight
        self.num_save_interval = num_save_interval
        self.print_freq = print_freq

        if is_writer:
            flog.debug('---- use tensorboard ---')
            from torch.utils.tensorboard import SummaryWriter
            c_time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
            _path = os.path.join(path_save_weight, c_time)
            os.makedirs(_path, exist_ok=True)
            self.tb_writer = SummaryWriter(_path)
        else:
            self.tb_writer = None

    def frun(self, start_epoch):
        t1 = time.time()
        for epoch in range(start_epoch, self.end_epoch + 1, 1):  # 从1开始
            save_val = None
            ''' ------------------- 训练代码  --------------------- '''
            if self.dataloader_train is not None and self.cfg.IS_TRAIN:
                loss_train_obj = self.ftrain(epoch, t1)
                save_val = loss_train_obj.avg

            ''' ------------------- 验证代码  --------------------- '''
            if self.dataloader_val is not None and self.cfg.IS_VAL and epoch % self.val_freq == 0:
                loss_val_obj = self.fval(epoch)

            if self.dataloader_train is not None and (epoch % self.num_save_interval) == 0:
                print('训练完成正在保存模型...')
                save_weight(
                    path_save=self.path_save_weight,
                    model=self.model,
                    name=self.save_weight_name,
                    loss=save_val,  # 这个正常是一样的有的
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    epoch=epoch)

    def fval(self, epoch):
        print('\n-------------------- 验证开始 开始 %s -------------------------' % epoch)
        losses_val_obj = AverageMeter()
        epoch_size = len(self.dataloader_val)

        self.model.eval()
        for i, reses in enumerate(self.dataloader_val):
            t0 = time.time()
            with torch.no_grad():
                loss_total, log_dict = self.fun_loss(reses)

                if i % (self.print_freq * self.cfg.BATCH_VAL) == 0:
                    show_loss_str = []
                    for k, v, in log_dict.items():
                        show_loss_str.append(
                            "{}: {:.4f} ||".format(k, v)
                        )

                    print('[Val %d/%d][Iter %d/%d/%d][lr %.6f]'
                          '[Loss: all %.2f || %s  time: %.2f]'
                          % (epoch, self.end_epoch, i + 1,
                             epoch_size, (epoch - 1) * epoch_size + i + 1,
                             self.optimizer.param_groups[0]['lr'],
                             losses_val_obj.avg,
                             str(show_loss_str),
                             time.time() - t0,
                             ),
                          flush=True
                          )

            if self.tb_writer is not None:
                iter = epoch_size * (epoch - 1) + i + 1
                # 主进程写入
                for k, v, in log_dict.items():
                    self.tb_writer.add_scalar('loss_iter/val-%s' % k, v, iter)

        return losses_val_obj

    def ftrain(self, epoch, t1):
        # 这里是 epoch
        # show_tb_writer = []
        loss_train_obj = AverageMeter()
        print('-------------------- fit_train 开始 %s -------------------------' % epoch)
        scaler = GradScaler(enabled=self.is_mixture_fit)
        epoch_size = len(self.dataloader_train)

        print('\n-------------------- 训练 开始 %s -------------------------' % epoch)
        for i, reses in enumerate(self.dataloader_train):
            t0 = time.time()
            if epoch < 2:
                now_lr = self.lr_val_base * pow((i + epoch * epoch_size) * 1. / (1 * epoch_size), 4)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = now_lr
            elif epoch == 1:
                now_lr = self.lr_val_base
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = now_lr

            with autocast(enabled=self.is_mixture_fit):
                if self.fun_loss is not None:
                    # 这里是回调
                    loss_total, log_dict = self.fun_loss(reses)
                else:
                    loss_total, log_dict = self.model(reses)

            loss_train_obj.update(loss_total.item())

            scaler.scale(loss_total).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            if i % (self.print_freq * self.cfg.BATCH_TRAIN) == 0:
                show_loss_str = []
                for k, v, in log_dict.items():
                    show_loss_str.append(
                        "{}: {:.4f} ||".format(k, v)
                    )

                print('[Train %d/%d][Iter %d/%d/%d][lr %.6f]'
                      '[Loss: all %.2f || %s  time: %.2f/%.2f]'
                      % (epoch, self.end_epoch, i + 1,
                         epoch_size, (epoch - 1) * epoch_size + i + 1,
                         self.optimizer.param_groups[0]['lr'],
                         loss_train_obj.avg,
                         str(show_loss_str),
                         t0 - t1,
                         time.time() - t1,
                         ),
                      flush=True
                      )

            if self.tb_writer is not None:
                # 主进程写入   不验证时写
                iter = epoch_size * (epoch - 1) + i + 1
                for k, v, in log_dict.items():
                    self.tb_writer.add_scalar('loss_iter/%s' % k, v, iter)
                self.tb_writer.add_scalar('loss_iter/lr', self.optimizer.param_groups[0]['lr'], iter)

            # 更新时间用于获取 data 时间
            t1 = time.time()

        return loss_train_obj


def save_weight(path_save, model, name, loss=None, optimizer=None, lr_scheduler=None, epoch=0, maps_val=None):
    if path_save and os.path.exists(path_save):
        sava_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'epoch': epoch}
        if maps_val is not None:
            if loss is not None:
                l = round(loss, 2)
            else:
                l = ''
            file_weight = os.path.join(path_save, (name + '-{}_{}_{}_{}.pth')
                                       .format(epoch + 1,
                                               l,
                                               'p' + str(round(maps_val[0] * 100, 1)),
                                               'r' + str(round(maps_val[1] * 100, 1)),
                                               ))
        else:
            file_weight = os.path.join(path_save, (name + '-{}_{}.pth').format(epoch + 1, round(loss, 3)))
        torch.save(sava_dict, file_weight)
        flog.info('保存成功 %s', file_weight)


def load_weight(file_weight, model, optimizer=None, lr_scheduler=None,
                device=torch.device('cpu'), is_mgpu=False, ffun=None):
    start_epoch = 1
    if file_weight and os.path.exists(file_weight):
        checkpoint = torch.load(file_weight, map_location=device)

        '''对多gpu的k进行修复'''
        # if 'model' in checkpoint:
        #     pretrained_dict_y = checkpoint['model']
        # else:
        #     pretrained_dict_y = checkpoint

        ''' debug '''
        # if ffun is not None:
        #     pretrained_dict = ffun(pretrained_dict_y)
        # else:
        #     pretrained_dict = pretrained_dict_y

        dd = {}

        # # 多GPU处理
        # ss = 'module.'
        # for k, v in pretrained_dict.items():
        #     if is_mgpu:
        #         if ss not in k:
        #             dd[ss + k] = v
        #         else:
        #             dd = pretrained_dict_y
        #             break
        #             # dd[k] = v
        #     else:
        #         dd[k.replace(ss, '')] = v

        '''重组权重'''
        # load_weights_dict = {k: v for k, v in weights_dict.items()
        #                      if model.state_dict()[k].numel() == v.numel()}

        keys_missing, keys_unexpected = model.load_state_dict(dd, strict=False)
        if len(keys_missing) > 0 or len(keys_unexpected):
            flog.error('missing_keys %s', keys_missing)  # 这个是 model 的属性
            flog.error('unexpected_keys %s', keys_unexpected)  # 这个是 pth 的属性
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler and checkpoint['lr_scheduler']:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1

        flog.warning('已加载 feadre 权重文件为 %s', file_weight)
    return start_epoch
