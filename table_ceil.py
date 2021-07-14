#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table ceil
@author: chineseocr
"""

import cv2
import numpy as np
from table_detect import table_detect
from table_line import table_line
from table_build import tableBuid,to_excel
from utils import minAreaRectbox, measure, eval_angle, draw_lines

class table:
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        self.img = img
        self.tableSize = tableSize
        self.tableLineSize = tableLineSize
        self.isTableDetect = isTableDetect
        self.isToExcel = isToExcel
        self.img_degree()
        self.table_boxes_detect()  ##表格定位
        self.table_ceil()  ##表格单元格定位

        self.table_build()

    def img_degree(self):
        img, degree = eval_angle(self.img, angleRange=[-15, 15])
        self.img = img
        self.degree = degree

    def table_boxes_detect(self):
        h, w = self.img.shape[:2]

        if self.isTableDetect:
            boxes, adBoxes, scores = table_detect(self.img, sc=self.tableSize, thresh=0.2, NMSthresh=0.3)
            if len(boxes) == 0:
                boxes = [[0, 0, w, h]]
                adBoxes = [[0, 0, w, h]]
                scores = [0]
        else:
            boxes = [[0, 0, w, h]]
            adBoxes = [[0, 0, w, h]]
            scores = [0]

        self.boxes = boxes
        self.adBoxes = adBoxes
        self.scores = scores

    def table_ceil(self):
        ###表格单元格
        n = len(self.adBoxes)
        self.tableCeilBoxes = []
        self.childImgs = []
        for i in range(n):
            xmin, ymin, xmax, ymax = [int(x) for x in self.adBoxes[i]]

            childImg = self.img[ymin:ymax, xmin:xmax]
            rowboxes, colboxes = table_line(childImg[..., ::-1], size=self.tableLineSize, hprob=0.5, vprob=0.5)
            tmp = np.zeros(self.img.shape[:2], dtype='uint8')
            tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)
            labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
            regions = measure.regionprops(labels)
            ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], True, True)
            ceilboxes = np.array(ceilboxes)
            ceilboxes[:, [0, 2, 4, 6]] += xmin
            ceilboxes[:, [1, 3, 5, 7]] += ymin
            self.tableCeilBoxes.extend(ceilboxes)
            self.childImgs.append(childImg)

    def table_build(self):
        tablebuild = tableBuid(self.tableCeilBoxes)
        cor = tablebuild.cor
        for line in cor:
            line['text'] = 'table-test'##ocr
        if self.isToExcel:
            workbook = to_excel(cor, workbook=None)
        else:
            workbook=None
        self.res = cor
        self.workbook = workbook


    def table_ocr(self):
        """use ocr and match ceil"""
        pass



if __name__ == '__main__':
    import argparse
    import os
    import time
    from utils import draw_boxes

    parser = argparse.ArgumentParser(description='tabel to excel demo')
    parser.add_argument('--isTableDetect', default=False, type=bool, help="是否先进行表格检测")
    parser.add_argument('--tableSize', default='416,416', type=str, help="表格检测输入size")
    parser.add_argument('--tableLineSize', default='1024,1024', type=str, help="表格直线输入size")
    parser.add_argument('--isToExcel', default=False, type=bool, help="是否输出到excel")
    parser.add_argument('--jpgPath', default='img/table-detect.jpg',type=str, help="测试图像地址")
    args = parser.parse_args()
    args.tableSize = [int(x) for x in args.tableSize.split(',')]
    args.tableLineSize = [int(x) for x in args.tableLineSize.split(',')]
    print(args)
    img = cv2.imread(args.jpgPath)
    t = time.time()
    tableDetect = table(img,tableSize=args.tableSize,
                        tableLineSize=args.tableLineSize,
                        isTableDetect=args.isTableDetect,
                        isToExcel=args.isToExcel
                        )
    tableCeilBoxes = tableDetect.tableCeilBoxes
    tableJson = tableDetect.res
    workbook =  tableDetect.workbook
    img = tableDetect.img
    tmp = np.zeros_like(img)
    img = draw_boxes(tmp, tableDetect.tableCeilBoxes, color=(255, 255, 255))
    print(time.time() - t)
    pngP = os.path.splitext(args.jpgPath)[0]+'ceil.png'
    cv2.imwrite(pngP, img)
    if workbook is not None:
        workbook.save(os.path.splitext(args.jpgPath)[0]+'.xlsx')
