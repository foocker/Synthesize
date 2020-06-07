import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# https://www.jianshu.com/p/64534f8eecc6
# 求两条线段的交点
def cross(a, b):
    '''平面向量的叉乘'''
    x1, y1 = a
    x2, y2 = b
    return x1 * y2 - x2 * y1


def line_cross(line1, line2):
    '''判断两条线段是否相交,并求交点'''
    a, b = line1
    c, d = line2
    # 两个三角形的面积同号或者其中一个为0（其中一条线段端点落在另一条线段上） ---> 不相交
    if cross(c - a, b - a) * cross(d - a, b - a) >= 0:
        return False
    if cross(b - c, d - c) * cross(a - c, d - c) >= 0:
        return False
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    k = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if k != 0:
        xp = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / k
        yp = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / k
    else:
        # 共线
        return False
    return xp, yp


def test1():
    lines = torch.randn((100,4)).view((-1,2,2))
    comb = combinations(lines,r =2 )
    plt.figure(figsize=(10,10))
    for line in lines:
        plt.plot(line[:,0],line[:,1],color = 'r')
    for line1,line2 in comb:
        r = line_cross(line1,line2)
        if r:
            plt.scatter(r[0],r[1], color='g')
    plt.show()

# test1()


# 整理点集顺序
def compare(a,b,center):
    '''
    对比a-center线段是在b-center线段的顺时针方向（True）还是逆时针方向（False）
    1. 通过叉乘积判断，积为负则a-center在b-center的逆时针方向，否则a-center在b-center的顺时针方向；
    2. 如果a,b,center三点共线，则按距离排列，距离center较远的作为顺时针位。

    原理：
    det = a x b = a * b * sin(<a,b>)
    其中<a,b>为a和b之间的夹角，意义为a逆时针旋转到b的位置所需转过的角度
    所以如果det为正，说明a可以逆时针转到b的位置，说明a在b的顺时针方向
    如果det为负，说明a可以顺时针转到b的位置，说明a在b的逆时针方向

    '''
    det = cross(a - center, b - center)
    if det > 0:
        return True
    elif det < 0:
        return False
    else:
        d_a = torch.sum((a - center) ** 2)
        d_b = torch.sum((b - center) ** 2)
        if d_a > d_b:
            return True
        else:
            return False


def quick_sort(box,left,right,center = None):
    '''快速排序'''
    if center is None:
        center = torch.mean(box,dim = 0)
    if left < right:
        q = partition(box,left,right,center)
        quick_sort(box,left,q - 1,center)
        quick_sort(box,q + 1,right,center)


def partition(box,left,right,center = None):
    '''辅助快排，使用最后一个元素将'''
    x = box[right]
    i = left - 1
    for j in range(left,right):
        if compare(x,box[j],center):
            i += 1
            temp = box[i].clone()
            box[i] = box[j]
            box[j] = temp
            # torch.Tensor不能使用下面的方式进行元素交换
            # box[i],box[j] = box[j],box[i]
    temp = box[i + 1].clone()
    box[i + 1] = box[right]
    box[right] = temp
    return i + 1


def test2():
    empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
    box = torch.rand((20, 2)) * 800
    cv2.polylines(empty, [box.data.numpy().astype(np.int32)], True, (0, 255, 0), 2)
    quick_sort(box, 0, len(box) - 1)
    cv2.polylines(empty, [box.data.numpy().astype(np.int32)], True, (255, 0, 0), 8)
    plt.imshow(empty)
    plt.show()


# test2()


# 判断点是否在多边形内
# 这个函数是用来求凸四边形交集的，因为凸四边形的交集图形的顶点由三部分构成：
#
# box1内部的box2的顶点；
# box2内部的box1的顶点；
# box1和box2的交点。
def inside(point,polygon):
    '''
    判断点是否在多边形内部
    原理：
    射线法
    从point作一条水平线，如果与polygon的焦点数量为奇数，则在polygon内，否则在polygon外
    为了排除特殊情况
    只有在线段的一个端点在射线下方，另一个端点在射线上方或者射线上的时候，才认为线段与射线相交
    '''
    x0,y0  = point
    # 做一条从point到多边形最左端位置的水平(y保持不变)射线
    left_line = torch.Tensor([[x0,y0],[torch.min(polygon,dim = 0)[0][0].item() - 1,y0]])
    lines = [[polygon[i],polygon[i+1]] for i in range(len(polygon) - 1)] + [[polygon[-1],polygon[0]]]
    ins = False
    for line in lines:
        (x1,y1),(x2,y2) = line
        if min(y1,y2) < y0 and max(y1,y2) >= y0:
            c = line_cross(left_line,line)
            if c and c[0] <= x0:
                ins = not ins
    return ins


def test3():
    empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
    box = torch.rand((20, 2)) * 800
    cv2.polylines(empty, [box.data.numpy().astype(np.int32)], True, (0, 255, 0), 2)
    quick_sort(box, 0, len(box) - 1)
    cv2.polylines(empty, [box.data.numpy().astype(np.int32)], True, (255, 0, 0), 8)

    points = torch.rand(800, 2) * 800
    for p_ in points:
        p = p_.clone().long()
        r = inside(p, box)
        if r:
            cv2.circle(empty, (p[0].item(), p[1].item()), 5, color=(0, 0, 0), thickness=5)
        else:
            cv2.circle(empty, (p[0].item(), p[1].item()), 5, color=(255, 0, 255), thickness=5)
    plt.imshow(empty)
    plt.show()


# test3()


# 求两个四边形的重叠区域
# !!!只适用于四边形的重叠区域只有一个的情况，例如两者都是凸四边形的情况
def intersection(box1,box2):
    '''
    判断两个框是否相交，如果相交，返回重叠区域的顶点
    1. 求box1在box2内部的点；
    2. 求box2在box1内部的点；
    3. 求box1和box2的交点；
    4. 所有点构成重叠区域的多边形点集；
    5. 顺时针排序
    '''
    quick_sort(box1,0,len(box1) - 1)
    quick_sort(box2,0,len(box2) - 1)
    # 求重叠区域
    # 整理成线段
    lines1 = [[box1[i],box1[i + 1]] for i in range(len(box1) - 1)] + [[box1[-1],box1[0]]]
    lines2 = [[box2[i],box2[i + 1]] for i in range(len(box2) - 1)] + [[box2[-1],box2[0]]]
    cross_points = []
    # 交点
    for l1 in lines1:
        for l2 in lines2:
            c = line_cross(l1,l2)
            if c:
                cross_points.append(torch.Tensor(c).view(1,-1))
    # 求box1在box2内部的点
    for b in box1:
        if inside(b,box2):
            cross_points.append(b.view(1,-1))
    for b in box2:
        if inside(b,box1):
            cross_points.append(b.view(1,-1))
    if len(cross_points) > 0:
        cross_points = torch.cat(cross_points,dim = 0)
        quick_sort(cross_points,0,len(cross_points) - 1)
        return cross_points
    else:
        return None


def test4():
    plt.figure(figsize=(18, 10))
    for i in range(4):
        box1 = torch.rand((4, 2)) * 800
        box2 = torch.rand((4, 2)) * 800
        empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
        quick_sort(box1, 0, len(box1) - 1)
        quick_sort(box2, 0, len(box2) - 1)
        cv2.polylines(empty, [box1.data.numpy().astype(np.int32)], True, (255, 0, 0), 4)
        cv2.polylines(empty, [box2.data.numpy().astype(np.int32)], True, (0, 255, 0), 4)
        cross_points = intersection(box1, box2)
        if cross_points is not None:
            cv2.polylines(empty, [cross_points.data.numpy().astype(np.int32)], True, (0, 0, 255), 4)
        plt.subplot(140 + i + 1)
        plt.imshow(empty)
    plt.show()


# test4()


# 计算多边形的面积
def polygon_area(polygon):
    '''
    求多边形面积,使用向量叉乘计算多边形面积，前提是多边形所有点按顺序排列
    '''
    lines = [[polygon[i],polygon[i+1]] for i in range(len(polygon) - 1)] + [[polygon[-1],polygon[0]]]
    s_polygon = 0.0
    for line in lines:
        a,b = line
        s_tri = cross(a,b)
        s_polygon += s_tri
    return s_polygon / 2


# 计算IOU
def intersection_of_union(box1,box2):
    '''
    iou = intersection(s_1,s_2) / (s_1 ＋ s_2 - intersection(s_1,s_2))
    '''
    quick_sort(box1,0,len(box1) - 1)
    quick_sort(box2,0,len(box2) - 1)
    s_box1 = torch.abs(polygon_area(box1))
    s_box2 = torch.abs(polygon_area(box2))
    cross_points = intersection(box1,box2)
    if cross_points is not None:
        s_cross = torch.abs(polygon_area(cross_points))
    else:
        s_cross = torch.Tensor([[0]])
    iou = s_cross / (s_box1 + s_box2 - s_cross)
    return iou


def test5():
    plt.figure(figsize=(18, 10))
    for i in range(4):
        box1 = torch.rand((4, 2)) * 800
        box2 = torch.rand((4, 2)) * 800
        empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
        quick_sort(box1, 0, len(box1) - 1)
        quick_sort(box2, 0, len(box2) - 1)
        #     s_box1 = torch.abs(polygon_area(box1))
        #     s_box2 = torch.abs(polygon_area(box2))
        cv2.polylines(empty, [box1.data.numpy().astype(np.int32)], True, (255, 0, 0), 4)
        cv2.polylines(empty, [box2.data.numpy().astype(np.int32)], True, (0, 255, 0), 4)
        cross_points = intersection(box1, box2)
        if cross_points is not None:
            cv2.polylines(empty, [cross_points.data.numpy().astype(np.int32)], True, (0, 0, 255), 4)
        #         s_cross = torch.abs(polygon_area(cross_points))
        #     else:
        #         s_cross = torch.Tensor([[0]])
        iou = intersection_of_union(box1, box2)
        print(iou.item())
        plt.subplot(140 + i + 1)
        plt.title("IOU : {}".format(iou.item()))
        plt.imshow(empty)
    plt.show()


# test5()


def nms_polygon(boxes,scores,score_thresh = 0.95,nms_thresh = 0.1):
    indices = torch.where(scores > score_thresh)[0]
    if len(indices) <= 1:
        return boxes[indices]
    boxes = boxes[indices]
    scores = scores[indices]
    keep_indices = []
    # 从大到小
    order = torch.argsort(scores).flip(dims = [0])
    while order.shape[0] > 0:
        i = order[0]
        keep_indices.append(i)
        not_overlaps = []
        for j in range(len(order)):
            if order[j] != i:
                iou = intersection_of_union(boxes[i],boxes[order[j]])
                if iou < nms_thresh:
                    not_overlaps.append(j)
        order = order[not_overlaps]
    keep_boxes = boxes[[i.item() for i in keep_indices]]
    return keep_boxes


def test6():
    boxes = torch.rand((10, 4, 2)) * 800
    empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
    for i in range(len(boxes)):
        quick_sort(boxes[i], 0, len(boxes[i]) - 1)
    cv2.polylines(empty, boxes.data.numpy().astype(np.int32), True, (0, 255, 0), 4)
    plt.subplot(121)
    plt.imshow(empty)
    scores = torch.arange(10) + 1
    keep_boxes = nms_polygon(boxes, scores)
    # print("keep indices",keep_indices,boxes.shape)
    # keep_boxes = boxes[[i.item() for i in keep_indices]]
    empty = (np.ones((800, 800, 3)) * 255).astype(np.uint8)
    cv2.polylines(empty, keep_boxes.data.numpy().astype(np.int32), True, (0, 255, 0), 4)
    plt.subplot(122)
    plt.imshow(empty)
    plt.show()


test6()