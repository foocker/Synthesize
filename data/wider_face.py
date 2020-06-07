import torch
from torch.utils import data
import cv2
import numpy as np


class WiderFaceDetection(data.Dataset):
    # for retinaface insight labeled txt, target: n*15
    def __init__(self, text_path, image_path,  preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.annotations = []
        with open(text_path, 'r') as f:
            lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst:
                    isFirst = False
                else:
                    label_copy = labels.copy()
                    self.annotations.append(label_copy)
                    labels.clear()
                line = line[1:].lstrip(' ')
                img_path = image_path + line
                # img_path = line.replace('# ', image_path)    # change
                self.imgs_path.append(img_path)
            else:
                label = line.split(' ')
                label = [float(l) for l in label]
                labels.append(label)
        self.annotations.append(labels)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        h, w, _ = img.shape

        labels = self.annotations[index]
        annotations = np.zeros((0, 15))    # concat all annotation
        if len(labels) == 0:
            return annotations
        for label in labels:
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]
            annotation[0, 1] = label[1]
            annotation[0, 2] = label[0] + label[2]
            annotation[0, 3] = label[1] + label[3]

            # landmark
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.imgs_path)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, img_anno in enumerate(sample):
            if torch.is_tensor(img_anno):
                imgs.append(img_anno)
            elif isinstance(img_anno, np.ndarray):
                annos = torch.from_numpy(img_anno).float()
                targets.append(annos)
    return torch.stack(imgs, 0), targets