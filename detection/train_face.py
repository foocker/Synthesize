from __future__ import print_function
import os
import torch
from torch import optim
from torch.utils import data
from data.wider_face import WiderFaceDetection, detection_collate
from data.transforms.data_augment import preproc
from .configs.face_config import cfg_mobile as cfg    # cfg_mobile, cfg_re50, cfg_facebox
from .losses.multibox_loss import MultiBoxLoss

import time
import datetime
import math


print('base cfg name is {}.'.format(cfg['name']))
rgb_mean = cfg['rgb_means']
num_classes = cfg['num_classes']
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = cfg['num_workers']
momentum = cfg['momentum']
weight_decay = cfg['weight_decay']
initial_lr = cfg['lr']
gamma = cfg['gamma']
training_label = cfg['training_label']
training_dataset = cfg['training_dataset']
save_weights = cfg['save_weights']
resume_epoch = cfg['resume_epoch']


net = cfg['net'](cfg=cfg, phase='train')
net = net.to('cuda')

if not os.path.exists(save_weights):
    os.mkdir(save_weights)

# resume

# distribute

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.3, False)

priorboxs = cfg['pribox'](cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorboxs.forward()
    priors = priors.to('cuda')    # change when distribute completed


def train():
    net.train()
    epoch = 0 + resume_epoch
    print('Loading Dataset....')

    dataset = WiderFaceDetection(training_label, training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if resume_epoch > 0:
        start_iter = resume_epoch * epoch_size
    else:
        start_iter = 0

    for it in range(start_iter, max_iter):
        if it % epoch_size == 0:
            epoch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers,
                                                  collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), os.path.join(save_weights, cfg['name'] + '_epoch_' + str(epoch) + '.pth'))
            epoch += 1

        load_start = time.time()
        if it in stepvalues:
            step_index += 1
        lr = adujst_learning_rate(optimizer, gamma, epoch, step_index, it, epoch_size)

        # load train data
        imgs, targets = next(epoch_iterator)    # batch or epoch_iterator?
        # print(targets)
        imgs = imgs.to('cuda')
        targets = [anno.to('cuda') for anno in targets]

        out = net(imgs)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        if loss_landm is not None:
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        else:
            loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_end = time.time()
        batch_time = load_end - load_start
        eta = int(batch_time * (max_iter - it))
        # print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || '
        #       'LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
        #     epoch, max_epoch, (it % epoch_size) + 1, epoch_size, it + 1, max_iter,
        #     loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f}  || '
              'LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
            epoch, max_epoch, (it % epoch_size) + 1, epoch_size, it + 1, max_iter,
            loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), os.path.join(save_weights, cfg['name'] + '_Final.pth'))


def adujst_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
     Adapted from PyTorch Imagenet example:
     https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
