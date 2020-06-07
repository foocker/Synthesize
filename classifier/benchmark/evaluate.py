import json
from torch.utils import data
from .inference import get_model
from .config import data_dir_test, batch_size, num_class, conf_matrix_path_json,\
     preds_results_json, classes
from .utils import confusion_matrix_eval, plot_confusion_matrix, handel_conf_matrix_path
from data.classifier_data import data_test_trainsfom, ImageFolderPath, ImageAnnotationPath, MultiLabelTransform
from classifier.benchmark import config as cfg


def evaluate():
    model = get_model()
    # dataset = datasets.ImageFolder(data_dir, data_train_val_trainsform)
    if cfg.datalabelmode == 'onelabel':
        dataset = ImageFolderPath(data_dir_test, data_test_trainsfom)
        batch_size = 1
    else:
        dataset = ImageAnnotationPath(data_dir_test, cfg.flagsfile, data_test_trainsfom, target_transform=MultiLabelTransform())
        batch_size = cfg.batch_size
    # dataset = train_val_dataset(cfg.data_dir, mode=cfg.datalabelmode, split=cfg.split_data_ration)
    # imgs = dataset.imgs
    # print(imgs[0])
    # print("dataset len:", len(dataset))
    data_loader = data.DataLoader(dataset, batch_size)
    conf_matrix, conf_matrix_path, results = confusion_matrix_eval(data_loader, model, num_class, mode='onelabel')
    with open(preds_results_json, 'w') as pre:
        json.dump(preds_results_json, pre, ensure_ascii=False)
    with open(conf_matrix_path_json, 'w') as f:
        json.dump(conf_matrix_path, f, ensure_ascii=False)
    # clas = ["Garbage", "Lines", "Stock", "Ad", "Shang"]
    # clas = ["Cigrarette_butts", "Fruit_flesh", "Fruit_peel", "Napkin"]
    # clas = list(dataset_class_to_idx.keys())
    clas = classes
    plot_confusion_matrix(conf_matrix.numpy(), clas)


# if __name__ == "__main__":
    # evaluate()
    # handel_conf_matrix_path()