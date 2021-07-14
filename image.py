#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
image
@author: chineseocr
"""

import base64
import json

import cv2
import numpy as np
import six
from PIL import Image


def plot_lines(img, lines, linetype=2):
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(tmp, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 0), linetype, lineType=cv2.LINE_AA)

    return Image.fromarray(tmp)


def base64_to_PIL(string):
    try:

        base64_data = base64.b64decode(string)
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except:
        return None


def read_json(p):
    with open(p) as f:
        jsonData = json.loads(f.read())
    shapes = jsonData.get('shapes')
    imageData = jsonData.get('imageData')
    lines = []
    labels = []
    for shape in shapes:
        lines.append(shape['points'])
        [x0, y0], [x1, y1] = shape['points']
        label = shape['label']
        if label == '0':
            if abs(y1 - y0) > 500:
                label = '1'
        elif label == '1':
            if abs(x1 - x0) > 500:
                label = '0'

        labels.append(label)
    img = base64_to_PIL(imageData)
    return img, lines, labels


from numpy import cos, sin, pi


def rotate(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def box_rotate(box, angle=0, imgH=0, imgW=0):
    """
    对坐标进行旋转 逆时针方向 0\90\180\270,
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    if angle == 90:
        x1_, y1_ = y2, imgW - x2
        x2_, y2_ = y3, imgW - x3
        x3_, y3_ = y4, imgW - x4
        x4_, y4_ = y1, imgW - x1

    elif angle == 180:
        x1_, y1_ = imgW - x3, imgH - y3
        x2_, y2_ = imgW - x4, imgH - y4
        x3_, y3_ = imgW - x1, imgH - y1
        x4_, y4_ = imgW - x2, imgH - y2

    elif angle == 270:
        x1_, y1_ = imgH - y4, x4
        x2_, y2_ = imgH - y1, x1
        x3_, y3_ = imgH - y2, x2
        x4_, y4_ = imgH - y3, x3
    else:
        x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = x1, y1, x2, y2, x3, y3, x4, y4

    return (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)


def angle_transpose(p, angle, w, h):
    x, y = p
    if angle == 90:
        x, y = y, w - x
    elif angle == 180:
        x, y = w - x, h - y
    elif angle == 270:
        x, y = h - y, x
    return x, y


def img_argument(img, lines, labels, size=(512, 512)):
    w, h = img.size
    if np.random.randint(0, 100) > 80:
        degree = np.random.uniform(-5, 5)
    else:
        degree = 0
    # degree = np.random.uniform(-5,5)
    newlines = []
    for line in lines:
        p1, p2 = line
        p1 = rotate(p1[0], p1[1], degree, w / 2, h / 2)
        p2 = rotate(p2[0], p2[1], degree, w / 2, h / 2)
        newlines.append([p1, p2])
    # img = img.rotate(-degree,center=(w/2,h/2),resample=Image.BILINEAR,fillcolor=(128,128,128))
    img = img.rotate(-degree, center=(w / 2, h / 2), resample=Image.BILINEAR)
    angle = np.random.choice([0, 90, 180, 270], 1)[0]
    newlables = []
    for i in range(len(newlines)):
        p1, p2 = newlines[i]
        p1 = angle_transpose(p1, angle, w, h)
        p2 = angle_transpose(p2, angle, w, h)
        newlines[i] = [p1, p2]
        if angle in [90, 270]:
            if labels[i] == '0':
                newlables.append('1')
            else:
                newlables.append('0')
        else:
            newlables.append(labels[i])

    if angle == 90:
        img = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        img = img.transpose(Image.ROTATE_270)

    return img, newlines, newlables


def fill_lines(img, lines, linetype=2):
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(tmp, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, linetype, lineType=cv2.LINE_AA)

    return tmp


def get_img_label(p, size, linetype=1):
    img, lines, labels = read_json(p)
    img, lines = img_resize(img, lines, target_size=512, max_size=1024)
    img, lines, labels = img_argument(img, lines, labels, size)
    img, lines, labels = get_random_data(img, lines, labels, size=size)

    lines = np.array(lines)
    labels = np.array(labels)
    labelImg0 = np.zeros(size[::-1], dtype='uint8')
    labelImg1 = np.zeros(size[::-1], dtype='uint8')

    ind = np.where(labels == '0')[0]
    labelImg0 = fill_lines(labelImg0, lines[ind], linetype=linetype)
    ind = np.where(labels == '1')[0]
    labelImg1 = fill_lines(labelImg1, lines[ind], linetype=linetype)

    labelY = np.zeros((size[1], size[0], 2), dtype='uint8')
    labelY[:, :, 0] = labelImg0
    labelY[:, :, 1] = labelImg1
    labelY = labelY > 0
    return np.array(img), lines, labelY


from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(image, lines, labels, size=(1024, 1024), jitter=.3, hue=.1, sat=1.5, val=1.5):
    '''random preprocessing for real-time data augmentation'''

    iw, ih = image.size

    # resize image
    w, h = size
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    # scale = rand(.2, 2)
    scale = rand(0.2, 3)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * nw / iw + dx, p1[1] * nh / ih + dy
        p2 = p2[0] * nw / iw + dx, p2[1] * nh / ih + dy
        lines[i] = [p1, p2]
    return image_data, lines, labels


def gen(paths, batchsize=2, linetype=2):
    num = len(paths)
    i = 0
    while True:
        # sizes = [512,512,512,512,640,1024] ##多尺度训练
        # size = np.random.choice(sizes,1)[0]
        size = 640

        X = np.zeros((batchsize, size, size, 3))
        Y = np.zeros((batchsize, size, size, 2))
        for j in range(batchsize):
            if i >= num:
                i = 0
                np.random.shuffle(paths)
            p = paths[i]
            i += 1

            # linetype=2
            img, lines, labelImg = get_img_label(p, size=(size, size), linetype=linetype)
            X[j] = img
            Y[j] = labelImg

        yield X, Y


def img_resize(im, lines, target_size=600, max_size=1500):
    w, h = im.size
    im_size_min = np.min(im.size)
    im_size_max = np.max(im.size)

    im_scale = float(target_size) / float(im_size_min)
    if max_size is not None:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

    im = im.resize((int(w * im_scale), int(h * im_scale)), Image.BICUBIC)
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * im_scale, p1[1] * im_scale
        p2 = p2[0] * im_scale, p2[1] * im_scale
        lines[i] = [p1, p2]

    return im, lines
