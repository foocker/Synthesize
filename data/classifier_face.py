from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


def tain_loader(cfg):
    assert isinstance(cfg.face_folder, list), 'face_folder should be list folder path'
    face_folders = []
    for fd in cfg.face_folder:
        face_folders.append(ImageFolder(fd, transforms=train_transform))
    folder_num = len(face_folders)
    if cfg.data_mode == 'concat' and folder_num > 1:
        for i in range(folder_num -1):
            tmp_class_num = face_folders[i][-1][1] + 1
            for idx, (img_p, label) in enumerate(face_folders[i+1].imgs):
                face_folders[i+1][idx] = (img_p, label + tmp_class_num)
        face_folder = ConcatDataset(face_folders)
        class_num = face_folders[1][-1][1] + 1
    else:
        face_folder = face_folders[0]
        class_num = face_folder[-1][1] + 1
    loader = DataLoader(face_folder, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    return loader, class_num


def val_loder(cfg):
    pass

