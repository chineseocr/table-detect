#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table ceil
@author: chineseocr
"""

import cv2
import numpy as np
from utils              import minAreaRectbox,measure,eval_angle,draw_lines
from table_detect import table_detect
from table_line   import table_line

class table:
   def __init__(self,img,tableSize = (416, 416),tableLineSize=(1024,1024),isTableDetect=False):
       self.img = img
       self.tableSize =tableSize
       self.tableLineSize = tableLineSize
       self.isTableDetect = isTableDetect
       self.img_degree()
       self.table_boxes_detect()##表格定位
       self.table_ceil()##表格单元格定位
       self.table_ocr()##表格文字识别
           
       
   def img_degree(self):
       img,degree =eval_angle(self.img,angleRange=[-15,15])
       self.img = img
       self.degree = degree
       
       
   def table_boxes_detect(self):
       h,w = self.img.shape[:2]
     
       if self.isTableDetect:
           boxes,adBoxes,scores=table_detect(self.img, sc=self.tableSize,thresh=0.2,NMSthresh=0.3)
           if len(boxes)==0:
               boxes = [[0,0,w,h]]
               adBoxes = [[0,0,w,h]]
               scores =[0]
       else:
           boxes = [[0,0,w,h]]
           adBoxes = [[0,0,w,h]]
           scores =[0]
        
           
       self.boxes = boxes
       self.adBoxes = adBoxes
       self.scores = scores
       
       
       
   def table_ceil(self):
       ###表格单元格
       n = len(self.adBoxes)
       self.tableCeilBoxes = []
       self.childImgs=[]
       for i in range(n):
           xmin,ymin,xmax,ymax = [int(x) for x in self.adBoxes[i]]
           
           childImg            = self.img[ymin:ymax,xmin:xmax]
           rowboxes,colboxes=table_line(childImg[...,::-1],size=self.tableLineSize,hprob=0.5,vprob=0.5)
           tmp = np.zeros(self.img.shape[:2],dtype='uint8')
           tmp = draw_lines(tmp,rowboxes+colboxes,color=255,lineW=2)
           labels=measure.label(tmp<255,connectivity=2)  #8连通区域标记
           regions = measure.regionprops(labels)
           ceilboxes= minAreaRectbox(regions,False,tmp.shape[1],tmp.shape[0],True,True)
           ceilboxes  = np.array(ceilboxes)
           ceilboxes[:,[0,2,4,6]]+=xmin
           ceilboxes[:,[1,3,5,7]]+=ymin
           self.tableCeilBoxes.extend(ceilboxes)
           self.childImgs.append(childImg)
           
   
   def table_ocr(self):
         pass
     
        
if __name__=='__main__':
    import time
    from utils import draw_boxes
    p = 'img/table-detect.jpg'
    img = cv2.imread(p)
    t =time.time()
    
    tableDetect = table(img)
    tableCeilBoxes = tableDetect.tableCeilBoxes
    img = tableDetect.img
    tmp = np.zeros_like(img)
    img = draw_boxes(tmp,tableDetect.tableCeilBoxes,color=(255,255,255))
    print(time.time()-t)
    cv2.imwrite('img/table-ceil.png',img)
    
               
               
               