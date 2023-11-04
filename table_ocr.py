#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用说明
1,获取源码
 1.1, https://github.com/chineseocr/table-detect
 1.2, 把本文件放到 table_ceil.py 所在目录
2,安装
 2.1,先看README.md,需要下载 检测table/cell 的预训练模型; 放到 models/目录下
 2.2,tesseract 与 easyocr 安装这里有
  https://blog.csdn.net/m0_66979647/article/details/125358293
  tesseract安装好之后, 找到 pytesseract.pytesseract.tesseract_cmd = './third-part/Tesseract-OCR/tesseract.exe'
  改为你实际 tesseract.exe 路径, 确保 pytesseract 能找到 tesseract.exe
 2.3, 在安装了本项目所需依赖包之后,还要安装新增的三个ocr库:
   pip install pytesseract easyocr cnocr
   如果运行时遇到 cv2.imshow error: (-2:Unspecified error) The function is not implemented.
   请参考 https://blog.csdn.net/chan1987818/article/details/129830435
3,运行
 3.1,把需要识别的图片放到img/目录下
 3.2,运行下面的命令. 有三种 OCR 库可供选择, 可对比哪个识别率更高.
  python table_ocr.py --jpgPath img/table-detect.jpg --isTableDetect True --isToExcel True --ocr PyTesseractOcrTable
  python table_ocr.py --jpgPath img/table-detect.jpg --isTableDetect True --isToExcel True --ocr CnOcrTable
  python table_ocr.py --jpgPath img/table-detect.jpg --isTableDetect True --isToExcel True --ocr EasyocrTable
  --jpgPath 后面的是图片路径
  --ocr 后面的是OCR库
  另外两个不变. 运行上述命令后, 会生成 xlsx 后追的Excel文件.

@author: JohnnySun
@time: Created on 2023/11/5 0:52
"""

import cv2
import numpy as np
from table_build import tableBuid, to_excel
from table_ceil import table


def text_format(text):
    return text.replace("\n", "").strip('|').strip().strip('|').strip()


class OcrTable(table):
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        super().__init__(img, tableSize, tableLineSize, isTableDetect, isToExcel)

    def table_build(self):
        tablebuild = tableBuid(self.tableCeilBoxes)
        cor = tablebuild.cor
        for line, rect in zip(cor, tablebuild.diagBoxes):
            line['text'] = text_format(self.cell_ocr(rect))
        if self.isToExcel:
            workbook = to_excel(cor, workbook=None)
        else:
            workbook = None
        self.res = cor
        self.workbook = workbook

    def cell_ocr(self, rect):
        return 'table-test'


class PyTesseractOcrTable(OcrTable):
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        import pytesseract  #
        pytesseract.pytesseract.tesseract_cmd = './third-part/Tesseract-OCR/tesseract.exe'
        self.pytesseract = pytesseract
        super().__init__(img, tableSize, tableLineSize, isTableDetect, isToExcel)

    def cell_ocr(self, rect):
        text = self.pytesseract.image_to_string(self.img[rect[1]:rect[3], rect[0]:rect[2]], lang='chi_sim+eng')
        print('cell', rect, text)
        return text


class CnOcrTable(OcrTable):
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        from cnocr import CnOcr
        self.CnOcr = CnOcr
        super().__init__(img, tableSize, tableLineSize, isTableDetect, isToExcel)

    def cell_ocr(self, rect):
        ocr = self.CnOcr()
        res = ocr.ocr_for_single_line(self.img[rect[1]:rect[3], rect[0]:rect[2]])
        print('cell', rect, res)
        return res['text']


class EasyocrTable(OcrTable):
    def __init__(self, img, tableSize=(416, 416), tableLineSize=(1024, 1024), isTableDetect=False, isToExcel=False):
        import easyocr
        self.easyocr = easyocr
        super().__init__(img, tableSize, tableLineSize, isTableDetect, isToExcel)

    def cell_ocr(self, rect):
        reader = self.easyocr.Reader(['ch_sim', 'en'])
        res = reader.readtext(self.img[rect[1]:rect[3], rect[0]:rect[2]], detail=0)
        print('cell', rect, res)
        if len(res) == 0:
            return ''
        else:
            return ' '.join(res)


if __name__ == '__main__':
    '''
    Usage:
        python table_ocr.py --jpgPath img/table-detect.jpg --isTableDetect True --isToExcel True
    '''
    import argparse
    import os
    import time
    from utils import draw_boxes

    parser = argparse.ArgumentParser(description='tabel to excel demo')
    parser.add_argument('--isTableDetect', default=False, type=bool, help="是否先进行表格检测")
    parser.add_argument('--tableSize', default='416,416', type=str, help="表格检测输入size")
    parser.add_argument('--tableLineSize', default='1024,1024', type=str, help="表格直线输入size")
    parser.add_argument('--isToExcel', default=False, type=bool, help="是否输出到excel")
    parser.add_argument('--jpgPath', default='img/table-detect.jpg', type=str, help="测试图像地址")
    parser.add_argument('--ocr', default='PyTesseractOcrTable', type=str, help="哪种OCR")
    args = parser.parse_args()
    args.tableSize = [int(x) for x in args.tableSize.split(',')]
    args.tableLineSize = [int(x) for x in args.tableLineSize.split(',')]
    print(args)
    img = cv2.imread(args.jpgPath)
    t = time.time()
    # 将string转为类名
    tableDetect = globals()[args.ocr](img, tableSize=args.tableSize,
                                      tableLineSize=args.tableLineSize,
                                      isTableDetect=args.isTableDetect,
                                      isToExcel=args.isToExcel
                                      )
    tableCeilBoxes = tableDetect.tableCeilBoxes
    tableJson = tableDetect.res
    workbook = tableDetect.workbook
    img = tableDetect.img
    tmp = np.zeros_like(img)
    img = draw_boxes(tmp, tableDetect.tableCeilBoxes, color=(255, 255, 255))
    print(time.time() - t)
    pngP = os.path.splitext(args.jpgPath)[0] + 'ceil.png'
    cv2.imwrite(pngP, img)
    if workbook is not None:
        workbook.save(os.path.splitext(args.jpgPath)[0] + '_' + type(tableDetect).__name__ + '.xlsx')
