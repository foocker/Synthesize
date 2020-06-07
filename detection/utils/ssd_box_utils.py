import torch
import numpy as np


def point_form(boxes):
    # c-->xmin, ymin, xmax, ymax
    if boxes.dim() == 3:
        boxes = boxes.squeeze(0)    # for general anchor
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)


def center_size(boxes):
    # xmin-->cx, cy, w, h
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)    # [A, B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union    # [A, B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_inter = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_inter / (area_a[:, np.newaxis] + area_b - area_inter)


def matrix_iof(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """
    Match each prior box with the ground truth box of the highest jaccard overlap, encode the
    bounding boxes, then return the matched indices corresponding to both confidence
    and locate preds.
        Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # overlaps = jaccard(truths, point_form(priors))
    overlaps = jaccard(truths, priors)
    # print(truths, priors[:2, :])

    # Bipartite Matching
    # [1, num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_index = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_index_filter = best_prior_index[valid_gt_idx, :]
    if best_prior_index_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        if landm_t is not None:
            landm_t[idx] = 0    # add
        return
    # [1, num_priors] best ground for each prior, not shape
    best_truth_overlap, best_truth_index = overlaps.max(0, keepdim=True)
    best_truth_index.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_index.squeeze_(1)
    best_prior_index_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ensure best prior?
    best_truth_overlap.index_fill_(0, best_prior_index_filter, 2)    # ?

    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_index.size(0)):
        best_truth_index[best_prior_index[j]] = j    # ?
    matches = truths[best_truth_index]    # will expand to [num_prior, 4]
    conf = labels[best_truth_index]
    conf[best_truth_overlap < threshold] = 0    # label as bg, as negative sample
    loc = encode(matches, priors, variances)

    if landms is not None and landm_t is not None:
        matches_landm = landms[best_truth_index]
        landm = encode_landm(matches_landm, priors, variances)
        landm_t[idx] = landm  # [num_priors, 10] encoded offsets to learn
    loc_t[idx] = loc
    conf_t[idx] = conf      # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)    # [num_priors, 4]


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)    # [num_priors, 5, 1]
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]    # [num_priors, 5, 2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape((g_cxcy.size(0), -1))    # [num_priors, 10]
    # may be give the box center, for five point is more good.
    return g_cxcy


def decode_landm(pre, priors, variances):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)

    return landms


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence losses across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel ==0:
        return keep
    x1 = boxes[:, 0]
    x2 = boxes[:, 1]
    y1 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = torch.mul(x2-x1, y2-y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]    # indices of the top-k largest vals
    xx1 = boxes.new()
    xx2 = boxes.new()
    yy1 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)    # size: len(idx)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next hightest score
        xx1 = torch.clamp(xx1, min=x1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as(xx2)
        h.resize_as(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check size of xx1 and xx2... after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=-0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)    # load remaining areas
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]

    return keep, count





