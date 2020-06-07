import torch
from torch import nn
# from ..utils.detbox import calc_iou
from ..utils.ssd_box_utils import center_size, jaccard, encode


class RetinaHeadLoss(nn.Module):
    def forward(self, predictions, anchors, targets):
        clas, regs = predictions
        alpha = 0.25
        gamma = 2.0
        batch_size = clas.shape[0]
        cla_losses, reg_losses = [], []
        anchor = anchors[0, :, :]
        # print(anchor.shape, anchor[1000:1010, :])
        anchor_centor_form = center_size(anchor)    # xmin-> cx, cy, w, h

        for idx in range(batch_size):
            cla = clas[idx, :, :]
            reg = regs[idx, :, :]
            target = targets[idx, :, :]
            target = target[target[:, 4] != -1]

            if target.shape[0] == 0:
                reg_losses.append(torch.tensor(0).float().to(anchors.device))
                cla_losses.append(torch.tensor(0).float().to(anchors.device))
                continue
            cla = torch.clamp(cla, 1e-4, 1.0 - 1e-4)
            iou = jaccard(anchor, target[:, :4])
            iou_max, iou_argmax = torch.max(iou, dim=1)
            # print(sum(iou_max > 0.5), sum(iou_max > 0.7), sum(iou_max > 0.9), target, iou_argmax.shape)
            # > 0.7 may aways exist, but 0.9 ... hard exit one

            # compute classification loss
            labels = torch.ones(cla.shape) * -1    # shape:(num_anchor, num_class)
            labels = labels.to(anchors.device)
            labels[torch.lt(iou_max, 0.4), :] = 0
            positive_indices = torch.ge(iou_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = target[iou_argmax, :]    # (num_anchor, 4+1)

            labels[positive_indices, :] = 0
            # setting 1 at real label's index
            labels[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(labels.shape) * alpha
            alpha_factor = alpha_factor.to(anchors.device)
            alpha_factor = torch.where(torch.eq(labels, 1), alpha_factor, 1.-alpha_factor)    # like label smooth
            focal_weight = torch.where(torch.eq(labels, 1), 1.-cla, cla)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(labels * torch.log(cla) + (1.0 - labels) * torch.log(1.0-cla))
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(labels, -1.0), cls_loss,  torch.zeros(cls_loss.shape).to(anchors.device))
            cla_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute location loss
            if positive_indices.sum() > 0:
                # should point form(dataloader is so) if use encode function
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_centor_form_choosed = anchor_centor_form[positive_indices, :]
                # center_gt = center_size(assigned_annotations)
                # clip widths to 1
                # center_gt[:, 2:] = torch.clamp(center_gt[:, 2:], min=1)
                # print(assigned_annotations.shape, anchor_centor_form.shape)
                encode_gt = encode(assigned_annotations[:, :4], anchor_centor_form_choosed, torch.Tensor([0.1, 0.2]))
                encode_gt = encode_gt.to(anchors.device)    # gt relative shift from anchor, and transform this shift

                reg_diff = torch.abs(encode_gt - reg[positive_indices, :])

                reg_loss = torch.where(
                    torch.le(reg_diff, 1.0/9.0),
                    0.5*9.0*torch.pow(reg_diff, 2),
                    reg_diff - 0.5/9.0
                )
                reg_losses.append(reg_loss.mean())
            else:
                reg_losses.append(torch.tensor(0).float().to(anchors.device))

        return torch.stack(cla_losses).mean(dim=0, keepdim=True), torch.stack(reg_losses).mean(dim=0, keepdim=True)

