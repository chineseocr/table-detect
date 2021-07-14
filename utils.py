#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
utils
@author: chineseocr
"""
import cv2
import numpy as np


def nms_box(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    ##nms box
    boxes = np.array(boxes)
    scores = np.array(scores)
    ind = scores > score_threshold
    boxes = boxes[ind]
    scores = scores[ind]

    def box_to_center(box):
        xmin, ymin, xmax, ymax = [round(float(x), 4) for x in box]
        w = xmax - xmin
        h = ymax - ymin
        return [round(xmin, 4), round(ymin, 4), round(w, 4), round(h, 4)]

    newBoxes = [box_to_center(box) for box in boxes]
    newscores = [round(float(x), 6) for x in scores]

    index = cv2.dnn.NMSBoxes(newBoxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index) > 0:
        index = index.reshape((-1,))
        return boxes[index], scores[index]
    else:
        return np.array([]), np.array([])


from scipy.ndimage import filters, interpolation
from numpy import amin, amax


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f)


def estimate_skew_angle(raw, angleRange=[-15, 15]):
    """
    估计图像文字偏转角度,
    angleRange:角度估计区间
    """
    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw - amin(raw)
    image = image / amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)
    # w,h = image.shape[1],image.shape[0]
    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w] - m[:h, :w] + 1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = amax(flat) - flat
    flat -= amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(angleRange[0], angleRange[1])
    estimates = []
    for a in angles:
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))

    _, a = max(estimates)
    return a


def eval_angle(img, angleRange=[-5, 5]):
    """
    估计图片文字的偏移角度
    """
    im = Image.fromarray(img)
    degree = estimate_skew_angle(np.array(im.convert('L')), angleRange=angleRange)
    im = im.rotate(degree, center=(im.size[0] / 2, im.size[1] / 2), expand=1, fillcolor=(255, 255, 255))
    img = np.array(im)
    return img, degree


def letterbox_image(image, size, fillValue=[128, 128, 128]):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    image_h, image_w = image.shape[:2]
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('tmp/test.png', resized_image[...,::-1])
    if fillValue is None:
        fillValue = [int(x.mean()) for x in cv2.split(np.array(image))]
    boxed_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    boxed_image[:] = fillValue
    boxed_image[:new_h, :new_w, :] = resized_image

    return boxed_image, new_w / image_w, new_h / image_h


from skimage import measure


def get_table_line(binimg, axis=0, lineW=10):
    ##获取表格线
    ##axis=0 横线
    ##axis=1 竖线
    labels = measure.label(binimg > 0, connectivity=2)  # 8连通区域标记
    regions = measure.regionprops(labels)
    if axis == 1:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > lineW]
    else:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > lineW]
    return lineboxes


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def adjust_lines(RowsLines, ColsLines, alph=50):
    ##调整line

    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x2, y2, x4, y4])

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newColsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newColsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newColsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newColsLines.append([x2, y2, x4, y4])

    return newRowsLines, newColsLines


def minAreaRect(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = image_location_sort_box(box)

    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2

    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2
    # degree,w,h,cx,cy = solve(box)
    # x1,y1,x2,y2,x3,y3,x4,y4 = box
    # return {'degree':degree,'w':w,'h':h,'cx':cx,'cy':cy}
    return [xmin, ymin, xmax, ymax]


def fit_line(p1, p2):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def point_line_cor(p, A, B, C):
    ##判断点与之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r


def line_to_line(points1, points2, alpha=10):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    A1, B1, C1 = fit_line((x1, y1), (x2, y2))
    A2, B2, C2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], A2, B2, C2)
    flag2 = point_line_cor([x2, y2], A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):

        x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
        y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
        p = (x, y)
        r0 = sqrt(p, (x1, y1))
        r1 = sqrt(p, (x2, y2))

        if min(r0, r1) < alpha:

            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]

    return points1


from scipy.spatial import distance as dist


def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    ---------------------
    作者：Tong_T
    来源：CSDN
    原文：https://blog.csdn.net/Tong_T/article/details/81907132
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def image_location_sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def xy_rotate_box(cx, cy, w, h, angle=0, degree=None, **args):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
    if degree is not None:
        angle = degree
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


from numpy import cos, sin


def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def minAreaRectbox(regions, flag=True, W=0, H=0, filtersmall=False, adjustBox=False):
    """
    多边形外接矩形
    """
    boxes = []
    for region in regions:
        rect = cv2.minAreaRect(region.coords[:, ::-1])

        box = cv2.boxPoints(rect)
        box = box.reshape((8,)).tolist()
        box = image_location_sort_box(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        angle, w, h, cx, cy = solve(box)
        if adjustBox:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w + 5, h + 5, angle=0, degree=None)

        if w > 32 and h > 32 and flag:
            if abs(angle / np.pi * 180) < 20:
                if filtersmall and w < 10 or h < 10:
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            if w * h < 0.5 * W * H:
                if filtersmall and w < 8 or h < 8:
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return boxes


from PIL import Image


def rectangle(img, boxes):
    tmp = np.copy(img)
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        cv2.rectangle(tmp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(tmp)


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]

    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA)

    return tmp


def draw_boxes(im, bboxes, color=(0, 0, 0)):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w, _ = im.shape

    for box in bboxes:
        if type(box) is dict:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(**box)
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]

        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1, lineType=cv2.LINE_AA)

    return tmp
