import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    # target is 4+1
    def __init__(self, root_dir, set_name='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_image(self, image_idx):
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def load_annotations(self, image_idx):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_idx], iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        # priorbox normalize box:
        # annotations /= w    # assume w == h

        return annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels  = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_idx):
        image = self.coco.loadImgs(self.image_ids[image_idx])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 2    # len(self.classes)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        h, w, _ = img.shape()
        annot = self.load_annotations(idx)
        targets = np.array(annot)
        bboxes = targets[:, :4] / max(w, h)
        labels = targets[:, 4]
        labels = np.array(labels, dtype=np.int)
        return torch.from_numpy(img), labels

    def __len__(self):
        return len(self.image_ids)


def coco_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    return torch.stack(imgs, 0), torch.FloatTensor(annot_padded)
