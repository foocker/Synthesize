import torch
import numpy as np
from .configs.config_eff import Args
from torch.utils.data import DataLoader
from .train_detection import prepare_device
from .models.efficientdet import EfficientDet
from ..data.coco import CocoDataset, coco_collate
from .utils.ssd_box_utils import jaccard
from ..data.transforms.augmentation import get_augumentation


args = Args()
args.data_root = ''
args.weights = ''

if args.dataset == 'VOC':
    pass
elif args.dataset == 'COCO':
    valid_dataset = CocoDataset(root=args.data_root,
                                  transform=get_augumentation(phase='valid'))
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.num_worker,
                              shuffle=False,
                              collate_fn=coco_collate(),
                              pin_memory=False)

if args.weights is not None:
    resume_path = str(args.weights)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(
        args.weights, map_location=lambda storage, loc: storage)
    num_class = checkpoint['num_class']
    network = checkpoint['network']
    model = EfficientDet(num_classes=num_class, network=network, is_training=False)
    model.load_state_dict(checkpoint['state_dict'])
device, device_ids = prepare_device(args.device)
model = model.to(device)

# if len(device_ids) > 1 :
#     model = torch.nn.DataParallel(model, device_ids=device_ids)


def evaluate_coco(threshold=0.5):
    import json
    from pycocotools import cocoeval
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []
        for idx, (images, annotations) in enumerate(valid_dataloader):
            images = images.to(device)
            annotations = annotations.to(device)
            scores, labels, boxes = model(images)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[: 1]
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break
                    image_result = {
                        'image_id': '',
                        'category_id': '',
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)
        if len(results) == 0:
            return None
        json.dump(results, open('{}_bbox_results.json'.format(valid_dataset.set_name), 'w'), indent=4)
        # load results in COCO evaluation tool
        coco_true = valid_dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(valid_dataset.set_name))

        # run COCO evaluation
        coco_eval = cocoeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))    # x-axis
    mpre = np.concatenate(([0.], precision, [0.]))    # y-axis

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):    # mpre.size - 1, mpre.size - 2, ..., 1
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_voc(iou_threshold=0.5):
    import tqdm
    model.eval()
    with torch.no_grad():
        all_detections = [[None for i in range(valid_dataset.__num_class__())] for j in range(len(valid_dataset))]
        all_annotations = [[None for i in range(valid_dataset.__num_class__())] for j in range(len(valid_dataset))]
        for idx, (images, annotations) in enumerate(tqdm(valid_dataloader)):
            images = images.to(device)
            annotations = annotations.to(device)
            scores, classification, transformed_anchors = model(images)
            if scores.shape[0] > 0:
                pred_annots = []
                for j in range(scores.shape[0]):
                    bbox = transformed_anchors[[j], :][0]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    idx_name = int(classification[[j]])
                    score = scores[[j]].cpu().numpy()
                    pred_annots.append([x1, y1, x2, y2, score, idx_name])
                pred_annots = np.vstack(pred_annots)
                for label in range(valid_dataset.__num_class__()):
                    all_detections[idx][label] = pred_annots[pred_annots[:, -1] == label, :-1]
            else:
                for label in range(valid_dataset.__num_class__()):
                    all_detections[idx][label] = np.zeros((0, 5))
            annotations = annotations[0].cpu().numpy()
            for label in range(valid_dataset.__num_class__()):
                all_annotations[idx][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('\t Start caculator mAP ...')
        average_precisions = {}

        for label in range(valid_dataset.__num_class__()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(valid_dataset.__num_class__()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = jaccard(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations

        print('\tmAP:')
        mAPS = []
        for label in range(valid_dataset.__num_class__()):
            label_name = valid_dataset.label_to_name(label)
            mAPS.append(average_precisions[label][0])
            print('{}: {}'.format(label_name, average_precisions[label][0]))
        print('total mAP: {}'.format(np.mean(mAPS)))
        return average_precisions

