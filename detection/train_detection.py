import os
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .configs.config_eff import EFFICIENTDET, Args
from .models.efficientdet import EfficientDet
from .losses.efficientdet_loss import RetinaHeadLoss
from data.transforms import get_augumentation
from data.coco import CocoDataset, coco_collate


args = Args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(
        device[0]) if n_gpu_use > 0 else 'cpu')

    return device, list_ids


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


if os.path.exists(args.resume):
    resume_path = str(args.resume)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(
        args.resume, map_location=lambda storage, loc: storage)
    args.num_classes = checkpoint['num_classes']
    args.network = checkpoint['network']

train_dataset = CocoDataset(root_dir=args.data_root, set_name='train2014', transform=get_augumentation(
    phase='train', width=EFFICIENTDET[args.network]['input_size'], height=EFFICIENTDET[args.network]['input_size']))


train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_worker,
                              shuffle=True,
                              collate_fn=coco_collate,
                              pin_memory=True)

model = EfficientDet(num_classes=args.num_classes,
                     network=args.network,
                     W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                     D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                     D_class=EFFICIENTDET[args.network]['D_class'],
                     )

if os.path.exists(args.resume):
    model.load_state_dict(checkpoint['state_dict'])
device, device_ids = prepare_device(args.device)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
criterion = RetinaHeadLoss()


def train():
    model.train()
    iteration = 1
    for ep in range(args.num_epoch):
        print("{} epoch: \t start training....".format(ep))
        start = time.time()
        total_loss = []
        optimizer.zero_grad()
        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.to(device)
            annotations = annotations.to(device)
            cla, reg, anchors = model(images)    # (cla, reg) predictions
            cla_loss, reg_loss = criterion((cla, reg), anchors, annotations)
            loss = cla_loss.mean() + reg_loss.mean()
            if bool(loss == 0):
                print('loss equal zero!')
                continue
            loss.backward()
            if (idx+1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
            total_loss.append(loss.item())
            if iteration % 100 == 0:
                print('{} iteration: training ...'.format(iteration))
                ans = {
                    'epoch': ep,
                    'iteration': iteration,
                    'cls_loss': cla_loss.item(),
                    'reg_loss': reg_loss.item(),
                    'mean_loss': np.mean(total_loss)
                }
                for key, value in ans.items():
                    print('    {:15s}: {}'.format(str(key), value))

            iteration += 1
        scheduler.step(np.mean(total_loss))
        result = {
            'time': time.time() - start,
            'loss': np.mean(total_loss)
        }
        for key, value in result.items():
            print('{:15s}: {}'.format(str(key), value))

        arch = type(model).__name__
        state = {
            'arch': arch,
            'num_class': args.num_classes,
            'network': args.network,
            'state_dict': get_state_dict(model)
        }

        if (ep+1) % args.save_freq_ep == 0:
            torch.save(state, '{}_{}_{}_{}.pth'.format(args.save_folder, args.dataset, args.network, ep))

    torch.save(state, '{}_Final_{}.pth'.format(args.save_folder, args.network))


