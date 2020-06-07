#coding=utf-8
#############################################
# mask nms　实现
# 2018.11.23 add
#############################################
import cv2
import numpy as np
import imutils
import copy

EPS=0.00001

def get_mask(box,mask):
    """根据box获取对应的掩膜"""
    tmp_mask=np.zeros(mask.shape,dtype="uint8")
    tmp=np.array(box.tolist(),dtype=np.int32).reshape(-1,2)
    cv2.fillPoly(tmp_mask, [tmp], (255))
    tmp_mask=cv2.bitwise_and(tmp_mask,mask)
    return tmp_mask,cv2.countNonZero(tmp_mask)


def comput_mmi(area_a,area_b,intersect):
    """
    计算MMI,2018.11.23 add
    :param mask_a: 实例文本a的mask的面积
    :param mask_b: 实例文本b的mask的面积
    :param intersect: 实例文本a和实例文本b的相交面积
    :return:
    """
    if area_a==0 or area_b==0:
        area_a+=EPS
        area_b+=EPS
        print("the area of text is 0")
    return max(float(intersect)/area_a,float(intersect)/area_b)


def mask_nms(dets, mask, thres=0.3):
    """
    mask nms 实现函数
    :param dets: 检测结果，是一个N*9的numpy,
    :param mask: 当前检测的mask
    :param thres: 检测的阈值
    """
    # 获取bbox及对应的score
    bbox_infos=dets[:,:8]
    scores=dets[:,8]

    keep=[]
    order=scores.argsort()[::-1]
    print("order:{}".format(order))
    nums=len(bbox_infos)
    suppressed=np.zeros((nums), dtype=np.int)
    print("lens:{}".format(nums))

    # 循环遍历
    for i in range(nums):
        idx=order[i]
        if suppressed[idx]==1:
            continue
        keep.append(idx)
        mask_a,area_a=get_mask(bbox_infos[idx],mask)
        for j in range(i,nums):
            idx_j=order[j]
            if suppressed[idx_j]==1:
                continue
            mask_b, area_b =get_mask(bbox_infos[idx_j],mask)

            # 获取两个文本的相交面积
            merge_mask=cv2.bitwise_and(mask_a,mask_b)
            area_intersect=cv2.countNonZero(merge_mask)

            #计算MMI
            mmi=comput_mmi(area_a,area_b,area_intersect)
            # print("area_a:{},area_b:{},inte:{},mmi:{}".format(area_a,area_b,area_intersect,mmi))

            if mmi >= thres:
                suppressed[idx_j] = 1

    return dets[keep]