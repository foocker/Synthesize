import copy
import torch
from checkpoint.checkpoint_base import Checkpointer

from .config import milestones

import numpy as np

def train_classify_fun(datainfo, labelmode, threshold, save_path, model, device, dataloaders, criterion, op,
                         scheduler=None, best_epoch_before=0, epoch_plus=6, ration=0.1, save_frequence=9):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.
    best_epoch = 0
    ep_loss = 0
    val_acc_hit = []
    CK = Checkpointer(model, save_path)
    # only save the best weights
    for ep in range(best_epoch_before, best_epoch_before+epoch_plus):
        print("epoch:{}".format(ep))
        if scheduler:
            lr = scheduler.get_lr()
            print('lr is :{}'.format(lr))
        for phase in ['train', 'val']:
            running_loss = 0.
            running_corrects = 0.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for imgs, labels, _ in dataloaders[phase]:
                # print(labels, 'lllllbales')
                # # imgs = imgs.to(device)
                # labels = labels.to(device)
                # # add some train loop tricks
                # # Random image cropping and patching
                # # 随机将4张图crop部分拼接成新图，并附加4个对应标签，也许能提升精度
                # beta = 0.3 # hyperparameter

                # # get the image size
                # I_x, I_y = imgs.size()[2:]

                # # draw a boundry position (w, h)
                # w = int(np.round(I_x * np.random.beta(beta, beta)))
                # h = int(np.round(I_y * np.random.beta(beta, beta)))

                # w_ = [w, I_x - w, w, I_x - w]
                # h_ = [h, h, I_y - h, I_y - h]

                # # select and crop four images
                # cropped_images = {}
                # c_ = {}
                # W_ = {}
                # for k in range(4):
                #     index = torch.randperm(imgs.size(0))
                #     x_k = np.random.randint(0, I_x - w_[k] + 1)
                #     y_k = np.random.randint(0, I_y - h_[k] + 1)
                #     cropped_images[k] = imgs[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                #     c_[k] = labels[index].to(device)
                #     W_[k] = w_[k] * h_[k] / (I_x * I_y)

                # # patch cropped images
                # patched_images = torch.cat(
                #     (torch.cat((cropped_images[0], cropped_images[1]), 2),
                #     torch.cat((cropped_images[2], cropped_images[3]), 2)),3)

                # patched_images = patched_images.to(device)
                # # get output
                # logits = model(patched_images)
                # # calculate loss and accuracy
                # losses = sum([W_[k] * criterion(logits, c_[k]) for k in range(4)])
                # acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])

                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                # print(logits.shape, '\n', torch.sigmoid(logits))
                losses = criterion(logits, labels)
                # print(labels, labels.shape, '\n', logits.shape)
                if phase == 'train':
                    op.zero_grad()
                    losses.backward()
                    op.step()
                    if scheduler and ep in milestones:    # milestones = [16, 20] in config file
                        scheduler.step()
                
                if labelmode == 'onelabel':
                    _, preds = torch.max(logits, 1)
                    # print('preds:', preds.shape, preds)
                    # print('where:', torch.sigmoid(logits) > 0.5)
                    running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
                    # temp_c = sum([W_[k]*torch.sum(preds.view(-1) == labels.view(-1)).item() for k in range(4)])
                    # running_corrects += temp_c
                    # print('labels:', labels, labels.view(-1), labels.shape, labels.view(-1).shape)
                else:
                    pred_multi = torch.sigmoid(logits) > threshold
                    # pred_multi = torch.topk(logits, )    # for all label k is not same
                    running_corrects += torch.sum(torch.all(pred_multi == labels, axis=1)).item()
                running_loss += losses.item() * imgs.size(0)    # when criterion mean
                

            ep_loss = running_loss / len(dataloaders[phase].dataset)
            ep_acc = running_corrects / len(dataloaders[phase].dataset)

            if phase == 'val':
                print("epoch:{}, val acc:{:.3f}".format(ep, ep_acc))
                val_acc_hit.append(ep_acc)
                if ep_acc > best_acc:
                    best_acc = ep_acc
                    best_epoch = ep
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print("epoch:{}, improve acc:{:.3f}".format(ep, best_acc))
                checkpoints = {
                    'epoch': best_epoch,
                    'model_state_dict': best_model_weights,
                    'optimizer_state_dict': op.state_dict(),
                    'losses': ep_loss,
                    'val_acc': best_acc,
                    'ratios': ration
                    }
                basename = '{}_epoch_{}_acc_{:.3f}_ration{}'.format(datainfo, best_epoch, best_acc, ration)
                # if (ep + 1) % save_frequence == 0:
                #     basename = '{}_epoch_{}_acc_{:.4f}_ration{}'.format(datainfo, ep, ep_acc, ration)
                #     CK.save(basename, checkpoints_)
            else:
                print("epoch:{}, train acc:{:.3f}".format(ep, ep_acc))
        print(checkpoints['epoch'], checkpoints['val_acc'])

    CK.save(basename, **checkpoints)
    return val_acc_hit
