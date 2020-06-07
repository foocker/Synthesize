import os
import cv2
from xpinyin import Pinyin
import shutil

import mmcv
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

file_format = ['png', 'jpg', 'jpeg', 'bmp', 'JPG']

p = Pinyin()

def rename_imgs(imagepath, extra_info=None, cp=False, new_dir=None):
    # extra_info for labeling added image
    for j, cla_name in enumerate(os.listdir(imagepath)):
        new_cla_name = p.get_initials(cla_name, u'') + '_' + str(j)
        for i, file_name in enumerate(os.listdir(os.path.join(imagepath, cla_name))):
            f_format = file_name.split('.')[-1]
            if f_format not in file_format:
                continue

            scr = os.path.join(imagepath, cla_name, file_name)
            dst = os.path.join(imagepath, cla_name, cla_name + '_' + str(i) + '_.' + f_format)
            if extra_info is not None:
                dst = os.path.join(imagepath, cla_name, cla_name + '_' + str(i) + extra_info + '_.' + f_format)
            if cp and new_dir is not None:
                new_dir_second = os.path.join(new_dir, new_cla_name)
                if not os.path.exists(new_dir_second):
                    os.makedirs(new_dir_second)
                new_filename = new_cla_name + '_' + str(i) + '_.' + f_format
                dst = os.path.join(new_dir_second, new_filename)

            os.rename(scr, dst)


def resize_imgs(imagepath, resized_shape):
    # shape will consider classify and detection
    for cla_name in os.listdir(imagepath):
        for file_name in os.listdir(os.path.join(imagepath, cla_name)):
            f_format = file_name.split('.')[-1]
            if f_format not in file_format:
                continue
            img = cv2.imread(os.path.join(imagepath, cla_name, file_name), cv2.IMREAD_COLOR )
            resized_img = cv2.resize(img, resized_shape, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imagepath, cla_name, file_name), resized_img)


def filter_img(dir_base, new_dir_base):
    for dir_r, _, files in os.walk(dir_base):
        if len(files) < 2:
            continue
        for img_f in files:
            cla_dir = os.path.basename(dir_r)
            img_p = os.path.join(dir_r, img_f)
            try:
                h, w, _ = cv2.imread(img_p).shape
            except:
                h, w = 0, 1
                continue
            if h > w:
                new_p = os.path.join(new_dir_base, cla_dir, img_f)
                if not os.path.exists(os.path.dirname(new_p)):
                    os.makedirs(os.path.dirname(new_p))
                shutil.copyfile(img_p, new_p)


one_bad = []

base_dir = ''
d0 = ''
d1 = 'one_good'
d2 = 'one_plus'
d3 = 'one_bad'

def creat_dir(base_dir, cla_dir):
    if  not os.path.exists(os.path.join(base_dir, cla_dir)):
        os.makedirs(os.path.join(base_dir, cla_dir))

creat_dir(base_dir, d1)
creat_dir(base_dir, d2)
creat_dir(base_dir, d3)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

    
def gamma_img(img, ga=0.3):
    if img.dtype == np.uint8:
        img = img / 255.
    return np.power(img, ga)

def linear_img(img, sacle=2):
    img_b = img * float(sacle)
    img_b[img_b>255] = 255
    img_b = np.round(img_b)
    img = img_b.astype(np.uint8)
    return img


def color_hist(img):
    b, g, r = cv2.split(img)
    b1 = cv2.equalizeHist(b)
    g1 = cv2.equalizeHist(g)
    r1 = cv2.equalizeHist(r)
    img = cv2.merge([b1, g1, r1])
    return img


def test_transform(base_d):
    i = 0
    for d, _ , files in os.walk(base_d):
        for f in files:
            img = cv2.imread(os.path.join(d, f))
            plt.subplot(121)
            plt.imshow(img)
            # img = np.float32(img)
            # img_photoed = photo(img)
            theta = 5 * random.randint(1, 4)
            img_rotated = rotate_bound(img, theta)
            img_ga = gamma_img(img_rotated, ga=1.6)
            # img_hist = color_hist(img_rotated)
            # img_linear = linear_img(img_rotated, sacle=2)
            plt.subplot(122)
            # plt.imshow(img_photoed)
            plt.imshow(img_ga)
            # plt.imshow(img_hist)
            # plt.imshow(img_linear)
            plt.show()
        i += 1
        if i == 10:
            break

def do_transform(base_d, save_d):
    # valid_names = []
    for d, _ , files in os.walk(base_d):
        for f in files:
            try:
                img = cv2.imread(os.path.join(d, f))
            except:
                print(d, f)
                continue
            if img is None:
                print(d, f)
                # valid_names.append(os.path.basename(d))
                # shutil.move(d, '')
                continue
            theta = 5 * random.randint(-4, 4)
            img_rotated = rotate_bound(img, theta)
            img_ga = gamma_img(img_rotated, ga=1.6) * 255
            img_ga = img_ga.astype(np.uint8)
            cla_name = os.path.basename(d)
            creat_dir(save_d, cla_name)
            save_name = f[:-4] + 'rotate_gamma' + f[-4:]
            cv2.imwrite(os.path.join(save_d, cla_name, save_name), img_ga)
    print('Done')
    # return valid_names


def merge_dir(dira, dirb):
    # merge a to b
    for d, _, fs in os.walk(dira):
        for f in fs:
            cla_name = os.path.basename(d)
            creat_dir(dirb, cla_name)
            os.rename(os.path.join(d, f), os.path.join(dirb, cla_name, f))
    print('merge done')

def check_all(dir_b):
    for d, _, fs in os.walk(dir_b):
        if len(fs) > 1:
            print(d)
            os.remove(d)
        else:
            continue
    print('check is done')


def reconstrcut_dir(dir_b):
    for dir_r, _, files in os.walk(dir_b):
        if len(files)==1:
            cla_dir = os.path.basename(dir_r)
            if cla_dir in one_bad:
                print(os.path.join(base_dir, d3))
                shutil.move(dir_r, os.path.join(base_dir, d3))
            else:
                print(os.path.join(base_dir, d1))
                shutil.move(dir_r, os.path.join(base_dir, d1))
        elif len(files) > 1:
            print(os.path.join(base_dir, d2))
            shutil.move(dir_r, os.path.join(base_dir, d2))
        else:
            continue
                
def remove_one(src, dst):
    for dirs, _, files in os.walk(src):
        if len(files):
            cla_name = os.path.basename(dirs)
            creat_dir(dst, cla_name)
            f = files[random.randint(0, len(files))]
            os.rename(os.path.join(dirs, f), os.path.join(dst, cla_name, f))


def make_same(dir_tr, dir_te):
    dirstr = os.listdir(dir_tr)
    dirste = os.listdir(dir_te)
    for i in dirste:
        if i in dirstr:
            del dirstr[dirstr.index(i)]
    return dirstr


if __name__ == '__main__':
    # imagepath = ''
    # new_dir = ''
    # new_dir_base = ''
    # extra_info = '_a'
    # resized_shape = (640, 640)
    # rename_imgs(imagepath, extra_info=None)
    # rename_imgs(imagepath, extra_info=None, cp=True, new_dir=new_dir)
    # resize_imgs(imagepath, resized_shape)
    # filter_img(new_dir, new_dir_base)
    # reconstrcut_dir(os.path.join(base_dir, d0))
    # img_p = ''
    # test_transform(img_p)
    # img_p = ''
    # img_g = ''
    # img_tr = ''
    # img_te = ''
    # img_p = ''

    # remove_one(img_p, img_g)
    # do_transform(img_tr, img_te)
    # merge_dir(img_p, img_te)
    # j = make_same(img_tr, img_te)
    # print(j)
    # check_all(img_tr)
    