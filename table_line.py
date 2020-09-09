#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table line detect
@author: chineseocr
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import LeakyReLU

def table_net(input_shape=(512, 512, 3),num_classes=1):
    inputs = Input(shape=input_shape)
    # 512
    use_bias=False
    down0a = Conv2D(16, (3, 3), padding='same',use_bias=use_bias)(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a = Conv2D(16, (3, 3), padding='same',use_bias=use_bias)(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same',use_bias=use_bias)(down0a_pool)
    down0 = BatchNormalization()(down0)
    
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(32, (3, 3), padding='same',use_bias=use_bias)(down0)
    down0 = BatchNormalization()(down0)
    down0 =LeakyReLU(alpha=0.1)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same',use_bias=use_bias)(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1 = Conv2D(64, (3, 3), padding='same',use_bias=use_bias)(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same',use_bias=use_bias)(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2 = Conv2D(128, (3, 3), padding='same',use_bias=use_bias)(down2)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same',use_bias=use_bias)(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(alpha=0.1)(down3)
    down3 = Conv2D(256, (3, 3), padding='same',use_bias=use_bias)(down3)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(alpha=0.1)(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same',use_bias=use_bias)(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU(alpha=0.1)(down4)
    down4 = Conv2D(512, (3, 3), padding='same',use_bias=use_bias)(down4)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU(alpha=0.1)(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same',use_bias=use_bias)(down4_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    center = Conv2D(1024, (3, 3), padding='same',use_bias=use_bias)(center)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same',use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    up4 = Conv2D(512, (3, 3), padding='same',use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    up4 = Conv2D(512, (3, 3), padding='same',use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same',use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    up3 = Conv2D(256, (3, 3), padding='same',use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    up3 = Conv2D(256, (3, 3), padding='same',use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same',use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    up2 = Conv2D(128, (3, 3), padding='same',use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    up2 = Conv2D(128, (3, 3), padding='same',use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same',use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    up1 = Conv2D(64, (3, 3), padding='same',use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    up1 = Conv2D(64, (3, 3), padding='same',use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same',use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same',use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same',use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same',use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    up0a = Conv2D(16, (3, 3), padding='same',use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    up0a = Conv2D(16, (3, 3), padding='same',use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model

from config import tableModeLinePath
from utils import letterbox_image,get_table_line,adjust_lines,line_to_line
import numpy as np
import cv2



model = table_net((None,None,3),2)
model.load_weights(tableModeLinePath)

def table_line(img,size=(512,512),hprob=0.5,vprob=0.5,row=50,col=30,alph=15):
    sizew,sizeh = size
    inputBlob,fx,fy = letterbox_image(img[...,::-1],(sizew,sizeh))
    pred = model.predict(np.array([np.array(inputBlob)/255.0]))
    pred = pred[0]
    vpred = pred[...,1]>vprob##竖线
    hpred = pred[...,0]>hprob##横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)
    colboxes = get_table_line(vpred,axis=1,lineW=col)
    rowboxes = get_table_line(hpred,axis=0,lineW=row)
    ccolbox = []
    crowlbox= []
    if len(rowboxes)>0:
       rowboxes = np.array(rowboxes)
       rowboxes[:,[0,2]]=rowboxes[:,[0,2]]/fx
       rowboxes[:,[1,3]]=rowboxes[:,[1,3]]/fy
       xmin = rowboxes[:,[0,2]].min()
       xmax = rowboxes[:,[0,2]].max()
       ymin = rowboxes[:,[1,3]].min()
       ymax = rowboxes[:,[1,3]].max()
       ccolbox = [[xmin,ymin,xmin,ymax],[xmax,ymin,xmax,ymax]]
       rowboxes = rowboxes.tolist()
       
    if len(colboxes)>0:
      colboxes = np.array(colboxes)
      colboxes[:,[0,2]]=colboxes[:,[0,2]]/fx
      colboxes[:,[1,3]]=colboxes[:,[1,3]]/fy
      
      xmin = colboxes[:,[0,2]].min()
      xmax = colboxes[:,[0,2]].max()
      ymin = colboxes[:,[1,3]].min()
      ymax = colboxes[:,[1,3]].max()
      colboxes = colboxes.tolist()
      crowlbox = [[xmin,ymin,xmax,ymin],[xmin,ymax,xmax,ymax]]
       
    rowboxes+=crowlbox
    colboxes+=ccolbox
    
    rboxes_row_,rboxes_col_=adjust_lines(rowboxes,colboxes,alph=alph)
    rowboxes+=rboxes_row_
    colboxes+=rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
       for j in range(ncol):
        rowboxes[i]=line_to_line(rowboxes[i],colboxes[j],10)
        colboxes[j]=line_to_line(colboxes[j],rowboxes[i],10)

    return rowboxes,colboxes


if __name__=='__main__':
    import time
    p = 'img/table-detect.jpg'
    from utils import draw_lines
    img = cv2.imread(p)
    t =time.time()
    rowboxes,colboxes=table_line(img[...,::-1],size=(512,512),hprob=0.5,vprob=0.5)
    img = draw_lines(img ,rowboxes+colboxes,color=(255,0,0),lineW=2)
    
    print(time.time()-t,len(rowboxes),len(colboxes))
    cv2.imwrite('img/table-line.png',img)