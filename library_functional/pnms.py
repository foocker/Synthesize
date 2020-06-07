# coding=utf-8
import numpy as np
from shapely.geometry import *


def py_cpu_pnms(dets, thresh):
    # 获取检测坐标点及对应的得分
    bbox = dets[:, :4]
    scores = dets[:, 4]

    # 这里文本的标注采用14个点，这里获取的是这14个点的偏移
    info_bbox = dets[:, 5:33]

    # 保存最终点坐标
    pts = []
    for i in range(dets.shape[0]):
        pts.append(
            [[int(bbox[i, 0]) + info_bbox[i, j], int(bbox[i, 1]) + info_bbox[i, j + 1]] for j in range(0, 28, 2)])

    areas = np.zeros(scores.shape)
    # 得分降序
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

    for il in range(len(pts)):
        # 当前点集组成多边形，并计算该多边形的面积
        poly = Polygon(pts[il])
        areas[il] = poly.area

        # 多剩余的进行遍历
        for jl in range(il, len(pts)):
            polyj = Polygon(pts[jl])

            # 计算两个多边形的交集，并计算对应的面积
            inS = poly.intersection(polyj)
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    # 下面做法和nms一样
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep