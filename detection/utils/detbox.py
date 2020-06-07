import numpy as np
import torch


def encode_box():
    # not need if all in loss forward
    pass


def decode_box(boxes, deltas, mean=None, std=None):
    # for inference
    if mean is None:
        mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
    if std is None:
        std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))

    widths = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

    return pred_boxes


def clipbox(boxes, img):
    # for inference
    batch_size, num_channels, height, width = img.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
    return boxes


def calc_iou(a, b):
    # same as ssd_box_utils.py jaccard(a, b)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU
