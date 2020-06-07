"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import os
import shutil
import time
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .efficient.efficientnet import EfficientNet
from .benchmark.config_efficient import cfg


best_acc1 = 0

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def main():
    main_worker(cfg)


def main_worker(cfg):
    global best_acc1

    if 'efficientnet' in cfg['arch']:   # NEW
        if cfg['pretrained']:
            model = EfficientNet.from_pretrained(cfg['arch'], num_classes=cfg['num_classes'])
            print("=> using pre-trained model '{}'".format(cfg['arch']))
        else:
            print("=> creating model '{}'".format(cfg['arch']))
            model = EfficientNet.from_name(cfg['arch'], override_params={'num_classes': cfg['num_classes']})      #  num_classes

    else:
        if cfg['pretrained']:
            print("=> using pre-trained model '{}'".format(cfg['arch']))
            model = models.__dict__[cfg['arch']](pretrained=True)    # num_classes?
        else:
            print("=> creating model '{}'".format(cfg['arch']))
            model = models.__dict__[cfg['arch']]()

    if device != 'cpu':
        model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), cfg['lr'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])

    # optionally resume from a checkpoint
    if cfg['resume']:
        if os.path.isfile(cfg['resume']):
            print("=> loading checkpoint '{}'".format(cfg['resume']))
            checkpoint = torch.load(cfg['resume'])
            cfg['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg['resume']))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(cfg['data'], 'train')
    valdir = os.path.join(cfg['data'], 'val')    # for val
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(cfg['image_size']),    # b5:456, b6:528, b7:600
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['workers'], pin_memory=True)

    if 'efficientnet' in cfg['arch']:
        # image_size = EfficientNet.get_image_size(cfg['arch'])    # 224-600,b1-b7
        image_size = cfg['image_size']
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', image_size)
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(cfg['image_size']),    # other backbone
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', cfg['image_size'])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['workers'], pin_memory=True)

    if cfg['evaluate']:
        res = validate(val_loader, model, criterion, cfg)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    for epoch in range(cfg['start_epoch'], cfg['epochs']):
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        print('epoch at {}, lr is{}'.format(epoch, lr))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, cfg)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, cfg)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint( {
            'epoch': epoch + 1,
            'arch': cfg['arch'],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, acc1, epoch, cfg['datainfo'])


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['print_freq'] == 0:
            progress.print(i)


def validate(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))    # top2
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg['print_freq'] == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, acc, epoch, datainfo):
    filename = '{}_valacc_{:.3f}_epoch_{}_.pth'.format(datainfo, acc, epoch)
    filename = os.path.join(cfg['save_weights'], filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# if __name__ == '__main__':
#     main()
