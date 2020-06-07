import json
import os
import copy
import cv2

import torch
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image

from classifier.benchmark import config as cfg

# when install will change .. to Synthesize

# IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

data_test_trainsfom = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.Resize(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5035, 0.5020, 0.5057], [0.2369, 0.2296, 0.2373])])


data_train_val_trainsform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.5, hue=0.05),
        transforms.Resize(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5035, 0.5020, 0.5057], [0.2369, 0.2296, 0.2373])
        ])



def train_val_dataset(data_dir, mode='onelabel', split=[0.1, 0.2, 0.3, 0.4]):
    assert mode in ['multilabel', 'onelabel']
    # dataset = datasets.ImageFolder(data_dir, data_train_val_trainsform)
    if mode == 'onelabel':
        dataset = ImageFolderPath(data_dir, data_train_val_trainsform)
    else:
        flagfile = 'flags.txt'
        dataset = ImageAnnotationPath(data_dir, flagfile, data_train_val_trainsform, target_transform=MultiLabelTransform())
    len_data = len(dataset)    # dataset.dataset?
    assert len_data > 10
    # raise Exception("size of data should > 10")
    test_size = [int(len_data * ra) for ra in split]  # min size > 10
    train_size = [len_data - t_s for t_s in test_size]
    train_test_dataset = [data.random_split(dataset, [tr_s, te_s]) for tr_s, te_s in
                          zip(train_size, test_size)]
    return train_test_dataset


def dataloaders_dict(batch_size, train_test_dataset, i):
    # get the spite rate[i] dataset :(train, val)
    return {x: data.DataLoader(train_test_dataset[i][index], batch_size=batch_size, shuffle=True,
                               num_workers=0) for x, index in zip(['train', 'val'], [0, 1])}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(dir, class_to_idx, image_format=['png', 'jpg', 'jpeg']):
    """
    for multillabel images classify, one image coressponding one json label file, 
    but image's format is not same, class_to_idx comes from flags.txt contain __ignore__ class
    """

    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            # __ignore__ dir is not exist at present
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                fprefix, fformat = fname.split('.')
                if fformat in image_format:
                    pimg = os.path.join(root, fname)
                    plabel = os.path.join(root, fprefix + '.json')
                    if os.path.exists(pimg) and os.path.exists(plabel):
                        item = (pimg, plabel)
                        images.append(item)
                    else:
                        print('{}_or_{} is not exits'.format(pimg, plabel))
                        continue
                else:
                    continue
    return images

                

class ImageFolderPath(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_loader, is_valid_file=None):
        super(ImageFolderPath, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          loader=loader,
                                          is_valid_file=is_valid_file)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

    def __len__(self):
        return len(self.samples)


class MultiLabelSparse(object):
    """
    sparse label from json file.
    return List[bool], like one-hot vector 
    """
    def __init__(self):
        pass

    def __call__(self, fjson):
        with open(fjson, 'r') as f:
            multilabel = json.load(f)["flags"]

        return list(multilabel.keys())


class MultiLabelTransform(object):
    """
    one hot and do some smooth tricks
    """
    def __init__(self):
        pass
    def __call__(self, multilabel):
        labels = torch.tensor(multilabel)
        labels = labels.unsqueeze(0)
        target = torch.zeros(labels.size(0), len(multilabel)).scatter_(1, labels, 1.)
        # may can add some tricks, such smooth
        return target



class ImageAnnotationPath(data.Dataset):
    def __init__(self, root, flagsfile, transform=None, 
                target_transform=None, loader=pil_loader):
        """
        add latter
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.flags = flagslist(os.path.join(root, flagsfile))
        self.class_to_idx = flagsdict(self.flags, mode='number')
        self.loader = loader
        self.images = make_dataset(root, self.class_to_idx)
        self.label_sparse = MultiLabelSparse()

    
    def __getitem__(self, index):
        ipath, lpath = self.images[index]
        sample = self.loader(ipath)
        target = self.label_sparse(lpath)
        if self.transform is not None:
            sample = self.transform(sample)    # in transform.Compose
        if self.target_transform is not None:
            target = self.target_transform(target)    # one hot and some trciks
        
        return sample, target, ipath


    def __len__(self):
        return len(self.images)


def flagslist(flagspath):
    with open(flagspath, 'r') as f:
        flags = [line.strip('\n') for line in f.readlines]

    return flags

def flagsdict(flags, mode='index'):
    assert mode in ['index', 'bool']
    flag_dict = dict()
    for flag, i in enumerate(sorted(flags)):
        if mode == 'number':
            flag_dict.update({flag:i})
        else:
            flag_dict.update({flag:False})

    return flag_dict


def reversedict(flag_dict):
    reversed_dict = {v:k for k, v in flag_dict.items()}
    return reversed_dict


def automultilabel(root, flagsfile, labelmapfile):
    """
    auto generate multilabel file(*.json), format as labelme, contains __ignore__ as zero class.
    labelmapfile is label_map.json, which map all given dir_name to their multi label index vector, 
    index comes from flags.txt(you must creat the flags.txt by the sorted order).
    """
    flags = flagslist(os.path.join(root, flagsfile))
    flags_dict_bool = flagsdict(flags, mode='bool')
    flags_dict_num = flagsdict(flags, mode='number')
    reversed_dict_num = reversedict(flags_dict_num)

    label_map = json.load(os.path.join(root, labelmapfile))
    ImageFormats = ('png', 'jpg', 'JPG', 'PNG', 'jpeg', 'JPEG')
    h, w = cfg.input_size
    dirs = os.listdir(root)
    for dir_cla in dirs:
        if os.path.isfile(dir_cla):
            continue
        for image_name in os.listdir(os.path.join(root, dir_cla)):
            if image_name.split('.')[-1] not in ImageFormats:
                continue
            # try:
            #     h, w, _ = cv2.imread(os.path.join(root, image_name)).shape
            # except:
            #     continue
            new_jsonname = image_name.split('.')[0] + '.json'
            with open(new_jsonname, 'w') as ff:
                label_dict = dict({"version": "4.2.9", "shapes": [], "imageData": None})
                tmp_flags = copy.copy(flags_dict_bool)
                for index in label_map[dir_cla]:
                    tmp_flags[reversed_dict_num[index]] = True
                label_dict['flags'] = tmp_flags
                label_dict["imagePath"] = image_name
                label_dict["imageHeight"] = h
                label_dict["imageWidth"] = w

                json.dump(label_dict, ff)


