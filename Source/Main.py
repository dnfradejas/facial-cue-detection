# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:06:41 2020

@author: Vince
"""
import cv2
import csv
import tensorflow as tf
from FacialFeatureExtraction import extractFacialFeatures
from FacialCuePrediction import Prediction
from ImageManip import scaleToWidth, showImage, scaleToHeight
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report

predictor = Prediction()
    

def detectFacialCues(filename, isImage=False):
    if not isImage:
        image = cv2.imread(filename)
        # scaledimage = scaleToWidth(image, 600)
        #showImage(scaledimage)
    else:
        image = filename
    left_eye, right_eye, mouth, coords = extractFacialFeatures(image)
    facemask = False
    cues = []
            
    
    if left_eye is not None and right_eye is not None and mouth is not None:
        cv2.imwrite('../Result/res_image.jpg', image)
        cv2.imwrite('../Result/res_lefteye.jpg', left_eye)
        cv2.imwrite('../Result/res_righteye.jpg', right_eye)
        cv2.imwrite('../Result/res_mouth.jpg', mouth)
        
        prediction = predictor.PredictDroopyEyelids('res_image.jpg', "Result")
        if prediction == 0:
            facemask = True
            
        prediction_left = predictor.PredictDarkCircles('res_lefteye.jpg', 'Result')
        prediction_right = predictor.PredictDarkCircles('res_righteye.jpg', 'Result')
        cues.append(prediction_left or prediction_right)
            
            
        prediction_left = predictor.PredictDroopyEyelids('res_lefteye.jpg', 'Result')
        prediction_right = predictor.PredictDroopyEyelids('res_righteye.jpg', 'Result')
        cues.append(prediction_left or prediction_right)
        
        prediction = predictor.PredictDroopyMouth('../Result/res_mouth.jpg', 'Result')
        cues.append(prediction)
            
            
        prediction_left = predictor.PredictEyeRedness('../Result/res_lefteye.jpg', 'Result')
        prediction_right = predictor.PredictEyeRedness('../Result/res_righteye.jpg', 'Result')
        cues.append(prediction_left or prediction_right)
            
            
        prediction_left = predictor.PredictSwollenEyelids('../Result/res_lefteye.jpg', 'Result')
        prediction_right = predictor.PredictSwollenEyelids('../Result/res_righteye.jpg', 'Result')
        cues.append(prediction_left or prediction_right)
    else:
        cues = [0,0,0,0,0]
            
            
    return cues


def performSkTesting():
   
    readcues = readActualCues()
    allactualcues = []
    allpredictedcues = []

    f = open('../predictions.csv', 'w')
    
    with f:
    
        writer = csv.writer(f)
        
        for i in range(1,181):
            str_i = str(i)
            if len(str_i) == 1:
                str_i = '00' + str_i
            elif len(str_i) == 2:
                str_i = '0' + str_i
                
            print("Image ", str_i)
            
            actualcues = []
            for j in readcues[i]:
                actualcues.append(int(j))
            allactualcues.append(actualcues)
            predictedcues = detectFacialCues('../TestImages/test-' + str_i + '.jpg')
            allpredictedcues.append(predictedcues)
            
            print("ACTUAL")
            print(actualcues)
            
            print("PREDICTED")
            print(str(predictedcues))
            writer.writerow(predictedcues)
            
            # img = cv2.imread('../TestImages/test-' + str_i + '.jpg')
            # showImage(img)
        y_true = np.array(allactualcues)
        y_pred = np.array(allpredictedcues)
        matrix = multilabel_confusion_matrix(y_true, y_pred)
        print(matrix)
        print(classification_report(y_true,y_pred))

def generateCroppedFeatures():   
    for i in range(1,181):
        if len(str(i)) == 3:
            i_str = str(i)
        elif len(str(i)) == 2:
            i_str = '0' + str(i)
        else:
            i_str = '00' + str(i)
        filename = "../TestImages/test-" + i_str + ".jpg"
        print(filename)
        image = cv2.imread(filename)
        image = scaleToWidth(image, 600)
        left_eye, right_eye, mouth, coords = extractFacialFeatures(image)
        
        if left_eye is not None and right_eye is not None and mouth is not None:
            left_eye = scaleToWidth(left_eye, 300)
            right_eye = scaleToWidth(right_eye, 300)
            cv2.imwrite( "../Cropped/lefteye-" + i_str + ".jpg", left_eye)
            cv2.imwrite( "../Cropped/righteye-" + i_str + ".jpg", right_eye)
            cv2.imwrite( "../Cropped/mouth-" + i_str + ".jpg", mouth)
            
def readActualCues():
    cues = []       
    with open('../cues.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            cues.append(row[0].split(',')[1:])
        #cues = cues[1:]
    return cues
    

def displayImage(filename):
    image = cv2.imread(filename)
    
    left_eye, right_eye, mouth, coords = extractFacialFeatures(image)
    cues = detectFacialCues(filename)
    
    for coord in coords:
        if coord is not None:
            cv2.rectangle(image, (coord[0],coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (255,255,255), 2)
    
    rescaledimage = scaleToHeight(image, 480)
    if len(rescaledimage[0]) > 640:
        rescaledimage = scaleToWidth(image,640)
    bg = cv2.imread('../Resource/bg.jpg')
    
    x = int(len(bg[0])/2 - len(rescaledimage[0])/2)
    y = int((len(bg)-80)/2 - len(rescaledimage)/2)
    h = len(rescaledimage)
    w = len(rescaledimage[0])
    bg[y:y+h,x:x+w] = rescaledimage 
    
    
    found = cv2.imread('../Resource/found.jpg')
    none = cv2.imread('../Resource/none.jpg')
    
    labeltext = [none, found]
    resultcoords = [[250,530],[250,570],[481,530],[481,570],[712,530]]
    
    h = 20
    w = 75
    
    for i in range(0,5):
        x = resultcoords[i][0]    
        y = resultcoords[i][1]
        bg[y:y+h,x:x+w] = labeltext[cues[i]]
    
    
    cv2.imshow("Facial Cue Detection", bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  

def displayAllImages():
    for i in range(94,181):
        if len(str(i)) == 3:
            i_str = str(i)
        elif len(str(i)) == 2:
            i_str = '0' + str(i)
        else:
            i_str = '00' + str(i)
        filename = "../TestImages/test-" + i_str + ".jpg"
        print(filename)
        
        #Function to display image with detected cues
        displayImage(filename)
        
        
        
def webCamFeed():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Facial Cue Detection")
    blank = cv2.imread('../Resource/blank.jpg')
    
    found = cv2.imread('../Resource/found.jpg')
    none = cv2.imread('../Resource/none.jpg')
    
    counter = 0
    count = False
    img = None
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        rescaledimage = frame
        bg = cv2.imread('../Resource/bg.jpg')
        x = int(len(bg[0])/2 - len(rescaledimage[0])/2)
        y = int((len(bg)-80)/2 - len(rescaledimage)/2)
        h = len(rescaledimage)
        w = len(rescaledimage[0])
        bg[y:y+h,x:x+w] = rescaledimage 
        k = cv2.waitKey(1)
        if count:
            counter += 1
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k%256 == 32:
            print("SPACE")
            # SPACE pressed
            count = True
            print(count)
        if k == ord('x'):
            # X pressed
            counter = 0
            count = False
        if counter > 0 and counter <= 90:
            cv2.putText(bg, "Subject detected", (100,60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        if counter > 0 and counter <= 30:
            cv2.putText(bg,"Capturing in 3", (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        if counter > 30 and counter <= 60:
            cv2.putText(bg,"Capturing in 2", (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        if counter > 60 and counter <= 90:
            cv2.putText(bg,"Capturing in 1", (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        if counter == 91:
            left_eye, right_eye, mouth, coords = extractFacialFeatures(rescaledimage)
            cues = detectFacialCues(rescaledimage, True)
            for coord in coords:
                if coord is not None:
                    cv2.rectangle(rescaledimage, (coord[0],coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (255,255,255), 2)
            bg[y:y+h,x:x+w] = rescaledimage 
            
            
            labeltext = [none, found]
            resultcoords = [[250,530],[250,570],[481,530],[481,570],[712,530]]
            
            h = 20
            w = 75
            
            for i in range(0,5):
                x = resultcoords[i][0]    
                y = resultcoords[i][1]
                bg[y:y+h,x:x+w] = labeltext[cues[i]]
            
            img = bg
        if counter <= 90:
            cv2.imshow("Facial Cue Detection", bg)
        elif counter == 91 :
            cv2.imshow("Facial Cue Detection", blank)
        else:
            cv2.imshow("Facial Cue Detection", img)
        if counter == 180:
            count = False
            counter = 0
    cam.release()
    
    cv2.destroyAllWindows()
    
  
    
def addText(image, text, org, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    fontScale = 0.4
    # Red color in BGR 
    # Using cv2.putText() method 
    image = cv2.putText(image, text, org, font, fontScale,  
                     color, thickness)  
    

if __name__ == "__main__":
    
    #Use to perform statistical testing using multi-label confusion matrix
    # performSkTesting()   
   
    
   # Use to display all test images in mock GUI
    # displayAllImages()
   
    
   #Use to display the webcam feed
    webCamFeed()
   
             
                
                
                
                
                
                
                
                
