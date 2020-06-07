# https://zhuanlan.zhihu.com/p/49481833
# https://zhuanlan.zhihu.com/p/50126479
import torch
import numpy as np


def py_cpu_nms(dets, thresh):
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # ------------------------------
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]                     # pred bbox top_x
    y1 = dets[:, 1]                     # pred bbox top_y
    x2 = dets[:, 2]                     # pred bbox bottom_x
    y2 = dets[:, 3]                     # pred bbox bottom_y
    scores = dets[:, 4]              # pred bbox cls score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    # pred bbox areas
    order = scores.argsort()[::-1]              # 对pred bbox按score做降序排序，对应step-2

    keep = []    # NMS后，保留的pred bbox
    while order.size > 0:
        i = order[0]          # top-1 score bbox
        keep.append(i)   # top-1 score的话，自然就保留了
        xx1 = np.maximum(x1[i], x1[order[1:]])   # top-1 bbox（score最大）与order中剩余bbox计算NMS
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)      # 无处不在的IoU计算~~~

        inds = np.where(ovr <= thresh)[0]     # 这个操作可以对代码断点调试理解下，结合step-3，我们希望剔除所有与当前top-1 bbox IoU > thresh的冗余bbox，那么保留下来的bbox，自然就是ovr <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        order = order[inds + 1]   # 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么 + 1？因为ind = 0就是这轮NMS的top-1，剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1 的映射，接下来就是step-4的循环

    return keep    # 最终NMS结果返回


def nms(boxes, scores, overlap=0.5, top_k=200):
    # Original author: Francisco Massa:
    # https://github.com/fmassa/object-detection.torch
    # Ported to PyTorch by Max deGroot (02/01/2017)
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object. ---- 这里面有一个细节，NMS仅用于测试阶段，为什么不用于训练阶段呢？可以评论留言下，我就不解释了，嘿嘿~~~
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)    # IoU初步准备
    v, idx = scores.sort(0)  # sort in ascending order，对应step-2，不过是升序操作，非降序
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals，依然是升序的结果
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:   # 对应step-4，若所有pred bbox都处理完毕，就可以结束循环啦~
        i = idx[-1]  # index of current largest val，top-1 score box，因为是升序的，所有返回index = -1的最后一个元素即可
        # keep.append(i)
        keep[count] = i
        count += 1    # 不仅记数NMS保留的bbox个数，也作为index存储bbox
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view，top-1已保存，不需要了~~~
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])   # 对应 np.maximum(x1[i], x1[order[1:]])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)    # clamp函数可以去查查，类似max、mini的操作
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        # 以下两步操作做了个优化，area已经计算好了，就可以直接根据idx读取结果了，area[i]同理，避免了不必要的冗余计算
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]     # 就是area(a) + area(b) - i
        IoU = inter/union  # store result in iou，# IoU来啦~~~
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]   # 这一轮NMS操作，IoU阈值小于overlap的idx，就是需要保留的bbox，其他的就直接忽略吧，并进行下一轮计算
    return keep, count


if __name__ == '__main__':
    dets = np.array([[100,120,170,200,0.98],
                     [20,40,80,90,0.99],
                     [20,38,82,88,0.96],
                     [200,380,282,488,0.9],
                     [19,38,75,91, 0.8]])

    py_cpu_nms(dets, 0.5)