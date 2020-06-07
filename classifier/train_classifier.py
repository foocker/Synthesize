from .benchmark.model import create_model, get_params_to_update, reusemodel, progressive_model
from .benchmark import config as cfg

from data.classifier_data import train_val_dataset, dataloaders_dict

import torch
import adabound
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from .benchmark.trainer import train_classify_fun

from data.transforms.labelsmooth import LabelSmoothingLoss
# from data.transforms.lsr import LSR

datasets = train_val_dataset(cfg.data_dir, mode=cfg.datalabelmode, split=cfg.split_data_ration)


label_w = [1.]* cfg.num_class
label_w[8] = 2.
label_w = torch.tensor(label_w, device=cfg.device)    # cfg.
criterion = nn.CrossEntropyLoss(weight=label_w) if cfg.datalabelmode == 'onelabel' else nn.BCEWithLogitsLoss()
# criterion = nn.KLDivLoss()
# criterion = LabelSmoothingLoss(cfg.num_class, smoothing=0.2)

# def get_num_class(cfg, mode='train_primary'):
#     assert mode in ['train_primary', 'pri_finetune', 'pro_finetune', 'progressive']
#     if mode == 'train_primary' or mode == 'pri_finetune':
#         return cfg.num_class_primary
#     if mode == 'progressive' or mode == 'pro_finetune':
#         return cfg.num_class
#     raise ValueError('mode is not supported')



def train(mode='train_primary'):
    assert mode in ['train_primary', 'pri_finetune', 'pro_finetune', 'progressive']
    labelmode = cfg.datalabelmode
    print('train mode is {}'.format(mode))
    for i in range(len(datasets)):
        num_class_use = cfg.num_class if mode == 'pro_finetune' else cfg.num_class_primary    # not clean
        model = create_model(num_class_use)
        if mode == 'train_primary':
            model = model
            best_epoch_before = 0
        elif mode.endswith('finetune'):
            model, best_epoch_before = reusemodel(model, cfg.weight_path)
        elif mode == 'progressive':
            model, best_epoch_before = progressive_model(model, cfg.weight_path, cfg.num_class)    # not new_class for evaluate create model
        else:
            raise ValueError('mode {} not supported!'.format(mode))
        model = model.to(cfg.device)
        params_to_update = get_params_to_update(model)
        # print(params_to_update)
        loader_dict = dataloaders_dict(cfg.batch_size, datasets, i)
        op = optim.SGD(params_to_update, lr=cfg.lr, momentum=cfg.lr_momentum)
        # op = adabound.AdaBound(params_to_update, lr=1e-3, final_lr=0.08)
        if cfg.lr_scheduler:
            scheduler = MultiStepLR(op, milestones=cfg.milestones, gamma=cfg.lr_gamma)
        else:
            scheduler = None

        ohist = train_classify_fun(cfg.datainfo, labelmode, cfg.multi_threshold, cfg.save_path, model, cfg.device, loader_dict, 
                                criterion, op,  scheduler=scheduler, best_epoch_before=best_epoch_before, epoch_plus=cfg.epoch_plus,
                                ration=cfg.split_data_ration[i])
        print(ohist)


# if __name__ == "__main__":
#     train()



