import cv2 as cv
import numpy as np
import os
import glob
import random
import time

from multiprocessing import Pool
import multiprocessing

def resize_keep(img, max_length):
    h, w,_ = img.shape
    if h > w:
        img = cv.resize(img, (int(max_length*w/h), int(max_length)), interpolation=cv.INTER_LINEAR)    # (w, h)
    else:
        img = cv.resize(img, (int(max_length), int(max_length*h/w)), interpolation=cv.INTER_LINEAR)

    return img


def gradcutmat(temp, bg, mode='template', ration=0.3):
    assert mode in ['detect', 'template']
    src = cv.imread(temp)
    
    if mode == 'detect':
        # model detect result, and crop
        # src = 
        pass
    h_s, w_s, _ = src.shape
    background = cv.imread(bg)
    # cv.imshow("bg", background)
    h, w, c = background.shape
    box_max_length = min(h, w) * ration 
    if max(h_s, w_s) > box_max_length:
        src = resize_keep(src, box_max_length)
        h_s, w_s, _ = src.shape
    
    bgback = np.zeros((h, w, c), dtype=np.uint8)
    mask =  np.zeros((h, w), dtype=np.uint8)
    locat_y = np.random.randint(1, h - h_s)
    locat_x = np.random.randint(1, w - w_s)
    bgback[locat_y:locat_y+h_s, locat_x:locat_x+w_s, :] = src

    rect = (locat_x, locat_y, w_s, h_s)
    bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组  13 * iterCount
    fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组  13 * iterCount
    cv.grabCut(bgback, mask, rect, bgdmodel, fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)

    # 提取前景和可能的前景区域
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    # cv.imwrite('background-mask_nn.jpg',mask2)    # not complete mask 
    result = cv.bitwise_and(bgback, bgback, mask=mask2)
    # cv.imwrite('result_grabcut.jpg', result)

    # 高斯模糊
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cv.dilate(mask2, se, mask2)
    mask2 = cv.GaussianBlur(mask2, (5, 5), 0)
    # cv.imshow('background-mask', mask2)

    # 虚化背景
    # background = cv.GaussianBlur(background, (0, 0), 15)
    mask2 = mask2/255.0
    a =  mask2[..., None]

    # 融合方法 com = a*fg + (1-a)*bg
    result = a* (bgback.astype(np.float32)) +(1 - a) * (background.astype(np.float32))
    # cv.imshow("result", result.astype(np.uint8))
    # cv.imwrite("result_gradcut_matting.jpg", result.astype(np.uint8))
    return result

    # cv.waitKey(0)
    # cv.destroyAllWindows()


# bgp = ''
# dsp = ''
# gradcutmat(bgp, dsp)

# x = glob.glob('')
def creat_cig_data(bg_resources, temp_resources, randomsampe=10):
    bg_r = glob.glob(bg_resources)
    for cla_name in os.listdir(temp_resources):
        for img_name in os.listdir(os.path.join(temp_resources, cla_name)):
            dsts = random.sample(bg_r, randomsampe)
            for fimg in dsts:
                try:
                    ration = random.choice([0.3, 0.35, 0.4, 0.42])
                    fuseimg = gradcutmat(os.path.join(temp_resources, cla_name, img_name), fimg, ration=ration)
                    fusedname = img_name[:-4] + '_' + os.path.basename(fimg)
                    cv.imwrite(os.path.join(temp_resources, cla_name, fusedname), fuseimg)
                except:
                    print(fusedname)
                    continue

# st = time.time()
# creat_cig_data('')
# print("cost time is:{}".format(time.time() - st))


def creat_cig_data_(bg_resources_list, temp_resources_list, randomsampe=6):
    # bg_r = glob.glob(bg_resources)
    # tmp_r = glob.glob(temp_resources)   # has second dir
    for cla_name_dir in temp_resources_list:
        for img_name in os.listdir(cla_name_dir):
            dsts = random.sample(bg_resources_list, randomsampe)
            for fimg in dsts:
                try:
                    ration = random.choice([0.3, 0.35, 0.4, 0.42])
                    fuseimg = gradcutmat(os.path.join(cla_name_dir, img_name), fimg, ration=ration)
                    fusedname = img_name[:-4] + '_' + os.path.basename(fimg)
                    cv.imwrite(os.path.join(cla_name_dir, fusedname), fuseimg)
                except:
                    print(fusedname)
                    continue

def muli_pthread(num_process, bg_resources, temp_resources):
    p = Pool(num_process)
    lines = len(temp_resources)
    part_line = lines // num_process
    part_list = [lines - part_line*i for i in range(num_process)][::-1]
    part_list.insert(0, 0)
    print(lines, part_line,  part_list)
    for i in range(num_process):
        print('run process:', i)
        p.apply_async(creat_cig_data_, args=(bg_resources, temp_resources[part_list[i]:part_list[i+1]]))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


def creat_cig_data_muilicore(temp_resources_list_one, randomsampe=6):
    # bg_r = glob.glob(bg_resources)
    # tmp_r = glob.glob(temp_resources)   # has second dir
    bg_resources_list = glob.glob('')
    cla_name_dir = temp_resources_list_one
    for img_name in os.listdir(cla_name_dir):
        dsts = random.sample(bg_resources_list, randomsampe)
        for fimg in dsts:
            try:
                ration = random.choice([0.35, 0.4, 0.42, 0.47])
                fuseimg = gradcutmat(os.path.join(cla_name_dir, img_name), fimg, ration=ration)
                fusedname = img_name[:-4] + '_' + os.path.basename(fimg)
                cv.imwrite(os.path.join(cla_name_dir, fusedname), fuseimg)
            except:
                print(fusedname)
                continue


def muli_core(num_process, temp_resources):
    # cores = multiprocessing.cpu_count()
    # print(cores)
    # p = Pool(processes==cores)
    with Pool(num_process) as p:
        p.map(creat_cig_data_muilicore, temp_resources)



if __name__ == '__main__':
    temp_resources = glob.glob('')
    bg_resources = glob.glob('')
    muli_pthread(5, bg_resources, temp_resources)
    # muli_core(5, temp_resources)

# src = cv.imread(')
# src = cv.resize(src, (0,0), fx=0.5, fy=0.5)
# print(src.shape, '1')
# r = cv.selectROI('input', src, False)  # 返回 (x_min, y_min, w, h)



# # roi区域
# # roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# img = src.copy()
# cv.rectangle(img, (int(r[0]), int(r[1])),(int(r[0])+int(r[2]), int(r[1])+ int(r[3])), (255, 0, 0), 2)
# # print(np.sum(img!=src))

# # 原图mask
# mask = np.zeros(src.shape[:2], dtype=np.uint8)

# # 矩形roi
# rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

# bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组  13 * iterCount
# fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组  13 * iterCount

# cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)
# print(src.shape, '2')

# # 提取前景和可能的前景区域
# mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
# # cv.imwrite('background-mask_nn.jpg',mask2)    # not complete mask 

# result = cv.bitwise_and(src,src,mask=mask2)
# # cv.imwrite('result_grabcut.jpg', result)
# # cv.imwrite('roi.jpg', roi)

# # Matting
# background = cv.imread("images/flower.png")

# h, w, ch = src.shape    # h, w
# background = cv.resize(background, (w, h))    # w, h
# # cv.imwrite("background.jpg", background)

# # 高斯模糊
# se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# cv.dilate(mask2, se, mask2)
# mask2 = cv.GaussianBlur(mask2, (5, 5), 0)
# cv.imshow('background-mask',mask2)
# # cv.imwrite('background-mask.jpg',mask2)


# # 虚化背景
# # background = cv.GaussianBlur(background, (0, 0), 15)
# mask2 = mask2/255.0
# a =  mask2[..., None]

# # 融合方法 com = a*fg + (1-a)*bg
# result = a* (src.astype(np.float32)) +(1 - a) * (background.astype(np.float32))

# cv.imshow("result", result.astype(np.uint8))
# # cv.imwrite("result_gradcut_matting.jpg", result.astype(np.uint8))

# cv.waitKey(0)
# cv.destroyAllWindows()