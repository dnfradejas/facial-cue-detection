# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:02:39 2020

@author: Vince
"""

import cv2

#interp = cv2.INTER_NEAREST 
#interp = cv2.INTER_LINEAR 
#interp = cv2.INTER_AREA 
#interp = cv2.INTER_LANCZOS4
interp = cv2.INTER_CUBIC

def showImage(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scaleByPercent(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = interp)
    return resized

def scaleToWidth(img, new_width):
    width = img.shape[1]
    height = img.shape[0]
    newwidth = new_width
    newheight = int(newwidth/width * height)
    dim = (newwidth, newheight)
    resized = cv2.resize(img, dim, interpolation = interp)
    return resized

def scaleToHeight(img, new_height):
    width = img.shape[1]
    height = img.shape[0]
    newheight = new_height
    newwidth = int(newheight/height * width)
    dim = (newwidth, newheight)
    resized = cv2.resize(img, dim, interpolation = interp)
    return resized

def scaleCrop(x,y,w,h, scale_percent):
    x_new = int(x - 0.5*scale_percent*w)
    y_new = int(y - 0.5*scale_percent*h)
    w_new = int(w + scale_percent*w)
    h_new = int(h + scale_percent*h)
    return x_new, y_new, w_new, h_new