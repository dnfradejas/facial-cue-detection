# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:38:55 2020

@author: Dayne Fradejas
"""
import os
import cv2
import tensorflow as tf
import math

class Prediction:
## This function is used for squared images 
    def __init__(self):
        os.chdir("../")
        self.IMAGE_DIR = os.getcwd() #get the current directory of images 
       
        os.chdir("Models")
        self.MODELS_DIR = os.getcwd() #get the current directory of models
   
        self.droopymouthcornersmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x2-CNN-DroopyMouth.model")
        self.darkcirclesmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x2-CNN-DarkCircles.model")
        self.eyerednessmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x2-CNN-EyeRedness.model")
        self.droopyeyelidsmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x2-CNN-DroopyEyelids.model")
        self.swolleneyelidsmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x2-CNN-SwollenEyelids.model")
        # self.facemasksmodel = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-FaceMask.model")
        
        
    def PrepareImage(self, filepath, width_x,length_y,image_type): 
        #image type 1 for GRAYSCALE, 3 for Colored Image
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array,(width_x,length_y))
        return new_array.reshape(-1,width_x,length_y,image_type)
    
    def PrepareColoredImageBox(self,filepath):
        IMG_X , IMG_Y = 300 , 300
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
    def PrepareBWImageBox(self,filepath):
        IMG_X , IMG_Y = 300 , 300
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
    def PrepareBWImage(self,filepath):      
        IMG_X , IMG_Y = 300 , 200
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
    def PrepareColredImage(self,filepath):
        IMG_X , IMG_Y = 300 , 200
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
    def PredictDroopyMouth(self,path,category):  
        prediction = self.droopymouthcornersmodel.predict(self.prepareDroopyMouth(os.path.join(self.IMAGE_DIR, category, path)))
        return math.ceil((prediction [0][0])) #0 is normal, and 1 is positive 
    
    def PredictDarkCircles(self,path,category):
        prediction = self.darkcirclesmodel.predict(self.prepareDarkCircles(os.path.join(self.IMAGE_DIR, category, path)))
        return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 

    def PredictEyeRedness(self,path,category):
        prediction = self.eyerednessmodel.predict(self.prepareEyeRedness(os.path.join(self.IMAGE_DIR, category, path)))
        return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
    def PredictDroopyEyelids(self,path,category):
        prediction = self.droopyeyelidsmodel.predict(self.prepareDroopyEyelids(os.path.join(self.IMAGE_DIR, category, path)))
        return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
    def PredictSwollenEyelids(self,path,category):
        prediction = self.swolleneyelidsmodel.predict(self.prepareSwollenEyelids(os.path.join(self.IMAGE_DIR, category, path)))
        return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
    def PredictFaceMask(self,path,category):
        prediction = self.facemasksmodel.predict(self.PrepareImage(os.path.join(self.IMAGE_DIR, category, path),150,150,3))
        (without_m, with_m) = prediction[0]
        if with_m > without_m: 
            return 1 #with facemask
        else:
            return 0 #without facemask
    
    def prepareDarkCircles(self,filepath):
        IMG_X , IMG_Y = 50 , 50
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,1)
        
    def prepareDroopyEyelids(self,filepath):
        IMG_X , IMG_Y = 50 , 50
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
    def prepareEyeRedness(self,filepath):
        IMG_X , IMG_Y = 50 , 50
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
    def prepareSwollenEyelids(self,filepath):
        IMG_X , IMG_Y = 50 , 50
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
    def prepareDroopyMouth(self,filepath):      
        IMG_X , IMG_Y = 64 , 48
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
        return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
# var = Prediction()
# print (var.PredictDroopyMouth("test-01.jpg" , "xd"))
# print (var.PredictDroopyMouth("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDroopyMouth("test-01.jpg" , "xd"))
# print (var.PredictEyeRedness("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-12.jpg","xd"))
# print (var.PredictDroopyEyelids("test-12.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDarkCircles("test-01.jpg" , "xd"))
# print (var.PredictDarkCircles("test-12.jpg" , "xd"))
# print (var.PredictDroopyMouth("test-01.jpg" , "xd"))
# print (var.PredictEyeRedness("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-12.jpg","xd"))
# print (var.PredictDroopyEyelids("test-12.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictDroopyEyelids("test-01.jpg","xd"))
# print (var.PredictSwollenEyelids("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))
# print (var.PredictFaceMask("test-01.jpg","xd"))






























# class Prediction:
# ## This function is used for squared images 
#     def __init__(self):
#         os.chdir("../Images")
#         self.IMAGE_DIR = os.getcwd() #get the current directory of images 
       
#         os.chdir("../Src/Models")
#         self.MODELS_DIR = os.getcwd() #get the current directory of models
#     def PrepareColoredImageBox(self,img):
#         IMG_X , IMG_Y = 300 , 300
#         new_array = cv2.resize(img,(IMG_X,IMG_Y))
#         return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
#     def PrepareBWImageBox(self,img):
#         IMG_X , IMG_Y = 300 , 300
#         new_array = cv2.resize(img,(IMG_X,IMG_Y))
#         new_array = cv2.cvtColor(new_array,cv2.COLOR_GRAY2BGR)
#         return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
#     def PrepareBWImage(self,filepath):      
#         IMG_X , IMG_Y = 300 , 200
#         img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#         new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
#         return new_array.reshape(-1,IMG_X,IMG_Y,1)
    
#     def PrepareColredImage(self,img):
#         IMG_X , IMG_Y = 300 , 200
#         new_array = cv2.resize(img,(IMG_X,IMG_Y))
#         new_array = cv2.cvtColor(new_array,cv2.COLOR_GRAY2BGR)
#         return new_array.reshape(-1,IMG_X,IMG_Y,3)
    
#     def PredictDroopyMouth(self,img):  
#         model = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-DroopyMouth.model")
#         prediction = model.predict(self.PrepareBWImage(img))
#         return math.ceil((prediction [0][0])) #0 is normal, and 1 is positive 
    
#     def PredictDarkCircles(self,img):
#         model = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-DarkCircles.model")
#         prediction = model.predict(self.PrepareBWImage(img))
#         return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 

#     def PredictEyeRedness(self,img):
#         model = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-EyeRedness.model")
#         prediction = model.predict(self.PrepareColoredImageBox(img))
#         return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
#     def PredictDroopyEyelids(self,img):
#         model = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-DroopyEyelid.model")
#         prediction = model.predict(self.PrepareBWImageBox(img))
#         return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
#     def PredictSwollenEyelids(self,img):
#         model = tf.keras.models.load_model(self.MODELS_DIR + "\\64x3-CNN-SwollenEyelids.model")
#         prediction = model.predict(self.PrepareColoredImageBox(img))
#         return math.ceil(prediction [0][0]) #0 is normal, and 1 is positive 
    
    