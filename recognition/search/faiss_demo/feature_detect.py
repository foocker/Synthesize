from os.path import isfile
import numpy as np
import logging
import cv2

from .config import *


# ------------- Get PHash
def calc_phash(gray_image):
    img = gray_image
    img = cv2.resize(img, (PHASH_X, PHASH_Y), interpolation=cv2.INTER_CUBIC)
    # create 2D array
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    # convert 2D array
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(PHASH_X, PHASH_Y)
    # convert to flat
    img_list = vis1.flatten()
    # mean value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = [np.float32(0) if i < avg else np.float32(1) for i in img_list]
    return np.matrix(avg_list).flatten()


# ------------- add Phash featrue
def adddPhash(gray_image, des):
    phash = calc_phash(gray_image)
    # merge phash and way
    n = des.shape[0]
    phash_mat = phash
    for _ in range(n - 1):
        phash_mat = np.vstack((phash_mat, phash))
    des = np.hstack((des, phash_mat))
    return des


def get_way(w_index):
    if w_index == 3:
        detector = cv2.xfeatures2d_SURF.create
        
    else:
        detector = eval('cv2.{}_create'.format(WAYS[w_index]))

    return detector(**PARAMETERS[w_index])

def read_img(img_f):
    if not isfile(img_f):
        logging.error('Image:{} does not exist'.format(img_f))
        return -1, None
    if isinstance(img_f, (np.ndarray, np.generic)):
        # not compelet on logic
        img = img_f
    else:
        try:
            img = cv2.imread(img_f)
        except:
            logging.error('Open Image:{} failed'.format(img_f))
            return -1, None

    if img is None:
        logging.error('Open Image:{} failed'.format(img_f))
        return -1, None
    # print('1',img.shape)
    img = cv2.resize(img, (NOR_X, NOR_Y))
    # print('2',img.shape)
    if img.ndim == 2:
        gray_img = img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return 0, gray_img


def way_feature(way, img_f):
    mode, gray_img = read_img(img_f)
    assert mode == 0, ('gray_img is None, check the read_img func')
    _, des = way.detectAndCompute(gray_img, None)    # kps, des
    # print(x, des)
    # print(des.shape)

    if isAddPhash:
        des = adddPhash(gray_img, des)

    features = np.matrix(des)

    return 0, features
    