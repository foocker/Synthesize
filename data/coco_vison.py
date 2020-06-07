import os
import cv2
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


font = cv2.FONT_HERSHEY_SIMPLEX

PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
"bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
"cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
"motorbike": 14, "person": 15, "pottedplant": 16,
"sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

CATEGORIES_INVERSE = {v: k for k, v in PRE_DEFINE_CATEGORIES.items()}


def resize_img_box(img, annotation, resized=(640, 640)):
    w, h = img.size
    w_sacle, h_scale = w / resized[0], h / resized[1]
    boxes = annotation['boxes']
    img = img.resize(resized)
    boxes[::2] /= w_sacle
    boxes[1::2] /= h_scale
    annotation['boxes'] = boxes
    return img, annotation


class COCODatasetVison(Dataset):
    # for torchvison dataset form: target is list(dict[tensor])
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)    #
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        annotation = dict()
        annotation["boxes"] = boxes
        annotation["labels"] = labels
        annotation["image_id"] = img_id
        annotation["area"] = areas
        annotation["iscrowd"] = iscrowd

        img, annotation = resize_img_box(img, annotation, resized=(640, 640))

        if self.transform is not None:
            img = self.transform(img)

        return img, annotation

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = list()
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loader(root='', annFile='', transform=None, batch_size=4, collate_fn=None):
    dataset = COCODatasetVison(root=root, annFile=annFile, transform=transform)
    data_laoder = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return data_laoder


def visualization(data_loader):
    for imgs, labels in data_loader:
        for i in range(len(imgs)):
            bboxes = []
            ids = []
            img = imgs[i]
            labels_ = labels[i]
            for label in labels_:
                bboxes.append([label['bbox'][0],
                               label['bbox'][1],
                               label['bbox'][0] + label['bbox'][2],
                               label['bbox'][1] + label['bbox'][3]
                               ])
                ids.append(label['category_id'])

            img = img.permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for box, id_ in zip(bboxes, ids):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                cv2.putText(img, text=CATEGORIES_INVERSE[id_], org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                            thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
            plt.imshow(img)
            plt.show()


# ------------------------------test data loader ----------------------------------
# path to your own data and coco file
# root = '/aidata/dataset/VOC/VOCdevkit/VOC2007/JPEGImages'
# annFile = '/aidata/dataset/VOC/VOCdevkit/VOC2007/AnnotationsJson/voc2007_to_coco.json'


# # select device (whether GPU or CPU)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# # DataLoader is iterable over Dataset
# i = 0
# for imgs, annotations in get_data_loader(**cfg):
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)
#     if i == 0:
#         break

# visualization(get_data_loader(**cfg))
# ------------------------------test data loader ----------------------------------
