# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:32:52 2020

@author: Vince
"""

import cv2
from ImageManip import showImage, scaleCrop, scaleToWidth, scaleToHeight
#import matplotlib.pyplot as plot


EYELOCATION_TOLERANCE = 0.5    #Higher than with respect to image size
EYESIZE_TOLERANCE = 0.5        #Bigger or smaller than with respect to the other eye
EYEVERTICAL_TOLERANCE = 0.4       #Higher than or lower than with respect to the other eye
EYEHORIZONTAL_TOLERANCE = 0.25     #To the right of with respect to the other eye

MOUTHLOCATION_TOLERANCE = 0.2    #With respect to the eyes
MOUTHSPACING_TOLERANCE = 1    #With respect to the eyes
MOUTHHORIZONTAL_TOLERANCE = 0.2  #With respect to the eyes

MOUTHVERTICAL_ESTIMATE = 0.5  #With respect to the eyes
MOUTHAREA_ESTIMATE = 1.75  #With respect to the eyes

EYESIZE_ESTIMATE = 0.8 #With respect to mouth width
EYEPOSITION_ESTIMATE = 0.2 #With respect to mouth height


def extractFacialFeatures(img):
    imgheight = len(img)
    imgWidth = len(img[0])
    
    #Load haar cascade classifier and perform detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    haar_eyes = eye_cascade.detectMultiScale(img, 1.05, 5)
    mouth_cascade = cv2.CascadeClassifier('../Classifiers/haarcascade_mouth.xml')
    haar_mouths = mouth_cascade.detectMultiScale(img, 1.05, 5)
    
    #Get only the objects detected in the upper half of the image
    eyes_roi = [] 
    for (x,y,w,h) in haar_eyes:
        if y < EYELOCATION_TOLERANCE * imgheight:
            eyes_roi.append([x, y, w, h])
    
    left_eye, right_eye, eyes_rect, eyecoords = extractEyes(img, eyes_roi)
    
#    print(eyes_rect)
#    

    mouth = None

    if left_eye is not None and right_eye is not None:
        #Load haar cascade classifier and perform detection
        
        #Get only the objects detected in the lower part of the eyes
        mouth_roi = []
        for (x,y,w,h) in haar_mouths:
            e1y = eyes_rect[0][1]
            e1h = eyes_rect[0][3]
            e2y = eyes_rect[1][1]
            e2h = eyes_rect[1][3]
            if y >  max(e1y + e1h + e1h * MOUTHLOCATION_TOLERANCE, e2y + e2h + e2h * MOUTHLOCATION_TOLERANCE):
                mouth_roi.append([x, y, w, h])
        
        mouth, mouth_rect = extractMouth(img, mouth_roi, eyes_rect)
    else:
        mouth, mouth_rect = extractMouth(img, haar_mouths)
        if mouth is not None:
            mx = mouth_rect[0]
            my = mouth_rect[1]
            mw = mouth_rect[2]
            mh = mouth_rect[3]
            
            
            eyesize = mw * EYESIZE_ESTIMATE
            spacing = mh * EYEPOSITION_ESTIMATE
            
            e1x = int(mx-mw*0.5+(mw-eyesize))
            e1y = int(my-eyesize-spacing)
            e1w = int(eyesize)
            e1h = int(eyesize)
            e2x = int(mx+mw*0.5)
            e2y = int(my-eyesize-spacing)
            e2w = int(eyesize)
            e2h = int(eyesize)
            
            
            if e1x < 0:
                e1x = 0
            if e2x + e2w > imgWidth:
                e2w = int(imgWidth - e2x)
            if e1y < 0 or e2y < 0:
                e1h = int(eyesize - abs(e1y))
                e2h = int(e1h)
                e1w = int(e1h)
                e2w = int(e1h)
                e1y = 0
                e2y = 0
            
            left_eye = img[e1y:e1y+e1h, e1x:e1x+e1w]
            right_eye = img[e2y:e2y+e2h, e2x:e2x+e2w]
            
            boxX = e1x
            boxY = e1y
            boxW = e2x + e2w - e1x
            boxH = e1h
            eyecoords = [boxX, boxY, boxW, boxH]
            
            # cv2.rectangle(img,(e1x,e1y), (e1x+e1w,e1y+e1h), (0,255,0), 2)
            # cv2.rectangle(img,(e2x,e2y), (e2x+e2w,e2y+e2h), (0,255,0), 2)
        # else:
        #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        #     haar_faces = face_cascade.detectMultiScale(img, 1.05, 5)
            
        #     faceresult = None
        #     for (x,y,w,h) in faces:
        #         if x < imgWidth/2 and x + w > imgWidth/2 and y < imgHeight/2 and y + h > imgHeight/2: 
        #             faceresult = 
            

    coords = [eyecoords,mouth_rect]
    
    #showImage(img)
    return left_eye, right_eye, mouth, coords





def extractMouth(img, roi, eye_details = None):
    mouth = "None :("
    
    mouth_results = []
        
    if eye_details is not None:
        elx = eye_details[0][0]
        ely = eye_details[0][1]
        elw = eye_details[0][2]
        elh = eye_details[0][3]
        erx = eye_details[1][0]
        ery = eye_details[1][1]
        erw = eye_details[1][2]
        erh = eye_details[1][3]
        
        leftthreshold_lower = elx + elw * MOUTHHORIZONTAL_TOLERANCE
        leftthreshold_upper = elx + elw
        rightthreshold_lower = erx
        rightthreshold_upper = erx + erw - erw * MOUTHHORIZONTAL_TOLERANCE
        

        for (x,y,w,h) in roi:
    #        print("Left Threshold L", leftthreshold_lower)
    #        print("Left Threshold U", leftthreshold_upper)
    #        print("Left Eye", x)
    #        print("Right Threshold L", rightthreshold_lower)
    #        print("Right Threshold U", rightthreshold_upper)
    #        print("Right Eye", x+w)
            #if x > leftthreshold_lower and x < leftthreshold_upper and x + w > rightthreshold_lower and x + w < rightthreshold_upper:
    #            print("PASS")\
                if ely + elh > ery + erh:
                    yn = int(ely + elh + elh * MOUTHSPACING_TOLERANCE)
                else:
                    yn = int(ery + erh + erh * MOUTHSPACING_TOLERANCE)
                if y < yn:
                    mouth_results.append([x,y,w,h])
                else:
                    pass          
    else:
        mouth_results = roi
        
        
        
    if len(mouth_results) == 1:  #If only one is detected, output the result
        xn, yn, wn, hn = scaleCrop(mouth_results[0][0], mouth_results[0][1], mouth_results[0][2], mouth_results[0][3], 0.15)   #Crop out a slightly larger area than the actual mouth detected
        # cv2.rectangle(img,(xn,yn),(xn+wn,yn+hn),(255,0,0),2)
        mouth = img[yn:yn+hn, xn:xn+wn]
        mouth_rect = [xn,yn,wn,hn]
    elif len(mouth_results) > 1 and eye_details is not None: #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Work In Progress
        min_index = 0
        for i in range(1,len(mouth_results)):
            if mouth_results[i][2] < mouth_results[min_index][2]:
                min_index = i
        
        xn, yn, wn, hn = mouth_results[min_index]
        if wn > elw and wn > erw and xn < erx and xn + wn > elx + elw:
            xn, yn, wn, hn = scaleCrop(mouth_results[0][0], mouth_results[0][1], mouth_results[0][2], mouth_results[0][3], 0.15)   #Crop out a slightly larger area than the actual mouth detected
            # cv2.rectangle(img,(xn,yn),(xn+wn,yn+hn),(255,0,0),2)
            mouth = img[yn:yn+hn, xn:xn+wn]
            mouth_rect = [xn,yn,wn,hn]
        else:
            for (x,y,w,h) in mouth_results:
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
                pass
            mouth = None
            mouth_rect = None
    elif len(mouth_results) < 1 and eye_details is not None:  #If none is detected, estimate a result
        xn = int(leftthreshold_lower)
        wn = int(rightthreshold_upper - leftthreshold_lower)
        hn = int(MOUTHAREA_ESTIMATE*min(elh,erh))
        if ely + elh > ery + erh:
            yn = int(ely + elh + elh * MOUTHVERTICAL_ESTIMATE)
        else:
            yn = int(ery + erh + erh * MOUTHVERTICAL_ESTIMATE)
        # cv2.rectangle(img,(xn,yn),(xn+wn,yn+hn),(0,255,0),2)
        mouth = img[yn:yn+hn, xn:xn+wn]
        mouth_rect = [xn,yn,wn,hn]
    else:
        mouth = None
        mouth_rect = None

    

    return mouth, mouth_rect


    
    
    
def extractEyes(img, roi):   
    #Containers for result
    pair = []
    throwout_pair = []
    
    if len(img) != 0:
        imgwidth = len(img[0])
        
    #If initial scan already produces two results, store as backup result
    n = len(roi)
    if n == 2:
        throwout_pair = roi
        
    #Set control variables    
    found = False
    i = 0

    while not found and i < n:
        j = i + 1
        while not found and j < n:
            ix = roi[i][0]
            iy = roi[i][1]
            iw = roi[i][2]
            ih = roi[i][3]
            jx = roi[j][0]
            jy = roi[j][1]
            jw = roi[j][2]
            jh = roi[j][3]
            uppersizethreshold = iw + iw * EYESIZE_TOLERANCE
            lowersizethreshold = iw - iw * EYESIZE_TOLERANCE
            if jw > lowersizethreshold and jw < uppersizethreshold:
                upperverticalthreshold = iy + iw * EYEVERTICAL_TOLERANCE
                lowerverticalthreshold = iy - iw * EYEVERTICAL_TOLERANCE
                if n == 2 and len(throwout_pair) == 0:
                    throwout_pair.append([ix, iy, iw, ih])
                    throwout_pair.append([jx, jy, jw, jh])
                if jy > lowerverticalthreshold and jy < upperverticalthreshold:
                    if n == 2 and len(throwout_pair) == 0:
                        throwout_pair.append([ix, iy, iw, ih])
                        throwout_pair.append([jx, jy, jw, jh])
                    spacing = 0
                    horizontalthreshold = 0
                    if ix < jx:
                        spacing = jx - (ix + iw)
                        horizontalthreshold = iw * EYEHORIZONTAL_TOLERANCE
                    elif jx < ix:
                        spacing = ix - (jx + jw)
                        horizontalthreshold = jw * EYEHORIZONTAL_TOLERANCE
                    if spacing > horizontalthreshold:
                        xcenter = imgwidth/2
                        if ix < jx and ix < xcenter and jx + jw > xcenter or jx < ix and jx < xcenter and ix + iw > xcenter:
                            if ix < jx and jx < ix + iw + 1.5*iw or jx < ix and ix < jx + jw + 1.5*jw:
                                pair.append([ix, iy, iw, ih])
                                pair.append([jx, jy, jw, jh])
                                found = True
            j += 1
        i += 1
        
    if len(pair) == 0 and len(throwout_pair) == 2:
        if throwout_pair[0][0] < throwout_pair[1][0]:
            left_to = throwout_pair[0] 
            right_to = throwout_pair[1] 
        else:
            left_to = throwout_pair[1] 
            right_to = throwout_pair[0] 
        if right_to[1] < left_to[1] + left_to[3] and right_to[1] + right_to[3] > left_to[1] :
            if right_to[0] < left_to[0] + left_to[2] + 2*left_to[2] and right_to[0] > left_to[0] + left_to[2]:
                pair = throwout_pair

    left_eye = None
    right_eye = None

    #If two eyes are detected
    if len(pair) == 2:
        e1x = pair[0][0]
        e1y = pair[0][1]
        e1w = pair[0][2]
        e1h = pair[0][3]
        e2x = pair[1][0]
        e2y = pair[1][1]
        e2w = pair[1][2]
        e2h = pair[1][3]
        
        #if eye 1 is on the left, proceed, if not, swap before proceeding
        if e1x < e2x:
            xn, yn, wn, hn = scaleCrop(e1x, e1y, e1w, e1h, 0.3) 
            if xn < 0:
                xn = 0
            if yn < 0:
                yn = 0         
            left_eye = img[yn:yn+hn, xn:xn+wn]
            xn, yn, wn, hn = scaleCrop(e2x, e2y, e2w, e2h, 0.3)
            if yn < 0:
                yn = 0
            if xn + wn > len(img[0]):
                xn -= (xn + wn - len(img[0]))
            right_eye = img[yn:yn+hn, xn:xn+wn]
            boxX = e1x
            boxY = min(e1y,e2y)
            boxW = e2x + e2w - e1x
            boxH = max(e1y+e1h,e2y+e2h) - min(e1y,e2y)
        else:
            xn, yn, wn, hn = scaleCrop(e2x, e2y, e2w, e2h, 0.3) 
            if xn < 0:
                xn = 0
            if yn < 0:
                yn = 0         
            left_eye = img[yn:yn+hn, xn:xn+wn]
            xn, yn, wn, hn = scaleCrop(e1x, e1y, e1w, e1h, 0.3)
            if yn < 0:
                yn = 0
            if xn + wn > len(img[0]):
                xn -= (xn + wn - len(img[0]))
            right_eye = img[yn:yn+hn, xn:xn+wn]
            temp = pair[0]
            pair[0] = pair[1]
            pair[1] = temp
            boxX = e2x
            boxY = min(e1y,e2y)
            boxW = e1x + e1w - e2x
            boxH = max(e1y+e1h,e2y+e2h) - min(e1y,e2y)
        eyecoord = [boxX, boxY, boxW, boxH]
    else:
        eyecoord = None
    
    for (x,y,w,h) in pair:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        pass
    
    return left_eye, right_eye, pair, eyecoord
    


# if __name__ == '__main__':
    
#     detected = 0
    
#     labels = ["DroopyEyelids", "DroopyMouthCorners", "EyeRedness", "SwollenEyelids", "DarkCircles"]
#     labels = ["SwollenEyelids"]
#     for label in labels:
#         for i in range(1,31):
#             counter = str(i)
#             print('Image ' + counter + ":")
#             stringname = ""
#             for i in range(0, 2-len(counter)):
#                 stringname = stringname + "0"
#             stringname += counter
#             image = cv2.imread('../TestImages/' + label + '/test-' + str(stringname) + '.jpg')
#             image = scaleToHeight(image, 600)
#             if extractFacialFeatures(image):
#                 detected += 1
#             showImage(image)
#         print(detected)
#             #extractv1 - Haar cascade only
    
    
    # for i in range(1,31):
    #     if len(str(i)) == 2:
    #         i_str = str(i)
    #     else:
    #         i_str = '0' + str(i)
            
    #     filename = "SwollenEyelids/test-" + i_str + ".jpg"
    #     image = cv2.imread(filename)
    #     left_eye, right_eye, mouth = extractFacialFeatures(image)
    #     showImage(image)
    
    #OUTPUT:
    #left_eye -> Image of the left eye OR None
    #right_eye -> Image of the right eye OR None
    #mouth -> Image of the mouth OR None
    
  