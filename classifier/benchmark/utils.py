import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from .config import conf_matrix_path_json, temp_data_path, classes, datainfo, multi_threshold, data_dir, data_dir_test

import json
import shutil
import os
from os.path import join

import cv2


def confusion_matrix(preds, labels, n_classes):
    # first get all preds and labels
    conf_matrix = torch.zeros(n_classes, n_classes)
    for la, pr in zip(labels, preds):
        conf_matrix[la, pr] += 1
    return conf_matrix


def confusion_matrix_batch(preds, labels, conf_matrix):
    # output = torch.randn(batch_size, n_classes)  # refer to output after softmax
    # target = torch.randint(0, n_classes, (batch_size,))  # labels
    # preds = torch.argmax(preds, 1)
    for la, pr in zip(labels, preds):
        conf_matrix[la, pr] += 1
    return conf_matrix


def confusion_matrix_batch_path(preds, labels, paths, conf_matrix_path):
    # conf_matrix_path:correspond conf_matrix, num_pred-->pred_path
    for la, pr, pa in zip(labels, preds, paths):
        if la != pr:
            conf_matrix_path['{}_{}'.format(la, pr)].append(pa)
    return conf_matrix_path


# dataset_class_to_idx = {'暴露垃圾': 0, '电线': 1, '物料堆积': 2, '小广告': 3, '游商':5}
# dataset_idx_to_class = {v:k for (k, v) in dataset_class_to_idx.items()}
# classes = ['暴露垃圾', '电线', '物料堆积', '小广告', '游商']
# classes = ["Cigrarette_butts", "Fruit_flesh", "Fruit_peel", "Napkin"]
# classes = ["Broken_brick", "Broken_road", "Doodle", "Exposed_garbage", "Drying_clothes", "Make_way",
#             "Manhole_cover_damage",  "Map", "Tobacco_advertising"]


def confusion_matrix_eval(data_loader, model, n_classes, mode='onelabel'):
    # batch_size = 1 .... 
    multi_label_map = {}    # key is index, value is one-hot tensor
    if torch.cuda.is_available():
        model = model.to("cuda")
    conf_matrix = torch.zeros(n_classes, n_classes)
    conf_matrix_path = {}
    results = {}    # just the result
    for i in range(n_classes):
        for j in range(n_classes):
            conf_matrix_path['{}_{}'.format(i, j)] = []
    with torch.no_grad():
        for imgs, labels, paths in data_loader:
            # print(labels, paths)
            imgs = imgs.to("cuda")
            labels = labels.to("cuda")
            label_mul_single = torch.zeros(labels.shape[0])    # transform one-hot label to its index
            preds_mul_single = torch.zeros(labels.shape[0])
            logits = model(imgs)
            if mode == 'onelabel':
                probality = torch.nn.functional.softmax(logits, dim=1)
                # print(probality, "------")
                preds = torch.argmax(probality, 1) 
            else:
                probality = torch.sigmoid(logits)   # pred_multi
                # pred_multi = probality > multi_threshold
                jl = 0
                jp = 0
                for vec_lab, vec_pred in zip(labels, probality):
                    k = int(vec_lab.sum().cpu().item())
                    va, indx = torch.topk(vec_pred, k)
                    temp = torch.tensor([1 if i in indx else 0 for i in range(n_classes)])    # contain __ignore__
                    # get label single firstly
                    for key, value in multi_label_map.items():
                        if torch.equal(value, vec_lab):
                            label_mul_single[j] = key
                            jl += 1
                        if torch.equal(value, temp):
                            preds_mul_single[jp] = key
                            jp += 1
                            break
                    # pred right will change to its cropressonding index otherwise to 0 which measn __ignore__
                    # if torch.equal(temp, vec_lab):
                    #         preds_mul_single[j] = label_mul_single[j]
                    # j += 1
                labels = label_mul_single.to("cuda")    # for code simple
                preds = preds_mul_single.to("cuda")

            # print(classes[preds.item()], '{:.3f}'.format(probality[0, preds.item()].item()))
            conf_matrix = confusion_matrix_batch(preds, labels, conf_matrix)
            # results[paths[0].split('/')[-1]] = [classes[preds.item()], '{:.3f}'.format(probality[0, preds.item()].item())]
            for i in range(labels.shape[0]):
                results[paths[i].split('/')[-1]] = [classes[preds[i].item()], '{:.3f}'.format(probality[0, preds[i].item()].item())]
            conf_matrix_path = confusion_matrix_batch_path(preds, labels, paths, conf_matrix_path)
            

    return conf_matrix, conf_matrix_path, results


def tp_fp_tn(conf_matrix, n_classes):
    # conf_matrix :tensor
    TP = conf_matrix.diag()
    recall = []
    precision = []
    sensitivity = []
    specificity = []
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity.append(TP[c] / (TP[c] + FN))
        specificity.append(TN / (TN + FP))
    return sensitivity, specificity


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype(float)    # add recall, precision
    c, r = cm.shape    # c == r
    c_sum, r_sum = cm.sum(axis=0), cm.sum(axis=1)
    cm = np.pad(cm, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    for i in range(c):
        cm[i, c] = cm[i, i] / r_sum[i]
        cm[c, i] = cm[i, i] / c_sum[i]

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes) +1)
    # colors = {cla:'b' for cla in classes}
    # colors.update({'Precision':'r', 'Recall':'r'}) 
    plt.xticks(tick_marks, classes + ['RECAL'], rotation=270, color='black')
    plt.yticks(tick_marks, classes + ['PRECISION'], color='black')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(c+1), range(r+1)):
        if i == c or j == r:
            fmt = '.2f'
        else:
            fmt = '.0f'
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label+Precision')
    plt.xlabel('Predicted label+ Recall')
    plt.savefig('{}_Confusion_matrix.png'.format(datainfo))
    # plt.show()


def handel_conf_matrix_path():
    with open(conf_matrix_path_json, 'r') as f:
        data = json.load(f)
    for k, vs in data.items():
        if not os.path.exists(temp_data_path + '/' + k):
            os.makedirs(temp_data_path + '/' + k)
        for v in vs:
            shutil.copyfile(v, temp_data_path + '/' + k + '/'+ v.split('/')[-1])    # linux


class DynamicAdjustData(object):
    def __init__(self):
        self.conf_json = conf_matrix_path_json
        self.temp_data = temp_data_path
        self.file_format = ['png', 'jpg', 'bmp', 'jpeg']

    def rename_imgs(self, imagepath, extra_info):
        # extra_info for labeling added image
        for cla_name in os.listdir(imagepath):
            for i, file_name in enumerate(os.listdir(os.path.join(imagepath, cla_name))):
                f_format = file_name.split('.')[-1]
                if f_format not in self.file_format:
                    continue

                scr = os.path.join(imagepath, cla_name, file_name)
                dst = os.path.join(imagepath, cla_name, cla_name + '_' + str(i) + extra_info + '_.' + f_format)
                os.rename(scr, dst)


    def resize_imgs(self, imagepath, resized_shape):
        # shape will consider classify and detection
        for cla_name in os.listdir(imagepath):
            for file_name in os.listdir(os.path.join(imagepath, cla_name)):
                f_format = file_name.split('.')[-1]
                if f_format not in self.file_format:
                    continue
                img = cv2.imread(os.path.join(imagepath, cla_name, file_name), cv2.IMREAD_COLOR )
                resized_img = cv2.resize(img, resized_shape, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(imagepath, cla_name, file_name), resized_img)


    def copy_conf_matrix_path(self):
        with open(conf_matrix_path_json, 'r') as f:
            data = json.load(f)
        for k, vs in data.items():
            if not os.path.exists(temp_data_path + '/' + k):
                os.makedirs(temp_data_path + '/' + k)
            for v in vs:
                shutil.copyfile(v, temp_data_path + '/' + k + '/' + v.split('/')[-1])  # linux

    # some efficient method to change file, data worker 

    def delete_checked_file(self, claname, checked_p):
        # delete from test dir
        assert claname in classes, '{} is not in given classes'.format(claname)
        if not os.path.isdir(checked_p):
            raise ValueError('{} must an valid dir'.format(checked_p))
        for img_f in os.listdir(checked_p):
            try:
                os.remove(join(data_dir_test, claname, img_f))
                os.remove(join(checked_p, img_f))
            except:
                print('delete_checked_file', img_f)
                continue
        print('delete test imgs which name in {}'.format(checked_p))
    

    def move_checked_ml(self, claname, checked_p, dest):
        assert claname in classes, '{} is not in given classes'.format(claname)
        if not os.path.exists(checked_p):
            raise ValueError('{} must an valid dir'.format(checked_p))
        if not os.path.exists(dest):
            os.mkdir(dest)
        for img_f in os.listdir(checked_p):
            try:
                os.rename(join(data_dir_test, claname, img_f), join(dest, img_f))
                os.remove(join(checked_p, img_f))
            except:
                print("move_checked_ml", img_f)
                continue
        print('move test imgs which name in {} to {}'.format(checked_p, dest))
    
    def sparse_dircla(self, dir_p):
        pass

    def move_checkedimg_test_train(self, claname, checked_p, ratio=0.8):
        # only one dir once
        # https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
        assert claname in classes, '{} is not in given classes'.format(claname)
        if not os.path.isdir(checked_p):
            raise ValueError('{} must an valid dir'.format(checked_p))
        all_wrong = len(os.listdir(checked_p))
        move_num = int(all_wrong * ratio)
        for img_f in os.listdir(checked_p)[:move_num]:
            try:
                os.rename(join(data_dir_test, claname, img_f), join(data_dir, claname, img_f))  # or replace
                os.remove(join(checked_p, img_f))
            except:
                print('move_checkedimg_test_train', img_f)
                continue
        print('move imgs from test to train which names in {} at ratio {} done!'.format(checked_p,  ratio))

    def data_static_info(self, root):
        pass