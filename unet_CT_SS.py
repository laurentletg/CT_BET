#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:18:46 2017

@author: m131199
"""

import os
import sys
import time
arg0 = sys.argv[0]
print(arg0)

# LLG added 
TAG = '3D_finetune_final_1' # will be added to folder name (see oLabel below line 34)
IMAGE_INPUT = '3D' # 2D or 3D
TRAIN_FROM_SCRATCH = False # set to True if you want to train from scratch
WEIGHTS_2D_ORIGINAL = 'unet_CT_SS_20171114_170726.h5'
WEIGHTS_3D_ORIGINAL = 'unet_CT_SS_3D_201843_163521.h5'
WEIGHTS_2D_UPDATED = None
WEIGHTS_3D_UPDATED = None


code_dir=os.getcwd()
hour = str(time.localtime()[3])
mins = str(time.localtime()[4])
sec = str(time.localtime()[5])
timeStamp = str(time.localtime()[0])+str(time.localtime()[1])+str(time.localtime()[2])+'_'+hour+mins+sec
               
sys.path.append(code_dir)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from auggen import AugmentationGenerator as dataGenerator
from model_CT_SS_LLG import Unet_CT_SS as genUnet


#===============output label & mode flags======================#
#*************set these flags before running*******************#
oLabel = arg0[arg0.rfind('/')+1:-3]+'_'+timeStamp + '_' + TAG
resultsFolder='results_folder'
pred_folder= 'predictions'
train=True
predict=False # run predictions for a specific weights.
#========================================================#
#create folders to save outputs

if  not os.path.lexists(os.path.join(code_dir,resultsFolder,oLabel)):
    os.mkdir(os.path.join(code_dir,resultsFolder,oLabel))
if  not os.path.lexists(os.path.join(code_dir,resultsFolder,oLabel,pred_folder)):
    os.mkdir(os.path.join(code_dir,resultsFolder,oLabel,pred_folder))
   
                      
#======================log file==========================#
logFile = open(os.path.join(code_dir,'results_folder',oLabel,'log_'+oLabel+'.txt'),'w') 
#========================================================#

dataAugmentation=False  
if dataAugmentation:         
    datagen = dataGenerator(rotation_z = 30,
            rotation_x = 0,
            rotation_y = 0,
            translation_xy = 5,
            translation_z = 0,
            scale_xy = 0.1,
            scale_z = 0,
            flip_h = True,
            flip_v = False)
    datagenPrams=datagen.__str__()
    afold=3
else:
    datagen=''
    datagenPrams=''
    afold=''
#===================set optimizer=====================#
# lr=1e-5
lr = 1e-6 # for fine-tuning
decay=1e-6
optimizer = 'adam'
#optimizer = SGD(lr=1e-4, momentum=0.9, decay=1e-9, nesterov=True)
#========================================================#

unetSS = genUnet(root_folder=code_dir, 
    image_folder = 'image_data',
    mask_folder = 'mask_data',
    save_folder= os.path.join(code_dir,resultsFolder,oLabel), 
    pred_folder=pred_folder,
    savePredMask=True,
    testLabelFlag=True,
    
    testMetricFlag=False, 
    dataAugmentation = dataAugmentation,
    logFileName='log_'+oLabel+'.txt',
    datagen=datagen,
    oLabel=oLabel,
    checkWeightFileName=oLabel+'.h5',
    afold=afold, 
    numEpochs=100,
    bs = 1, 
    nb_classes=2,
    sC=2, #saved class
    img_row=512,img_col=512,channel=1,
    classifier = 'softmax',
    optimizer =optimizer,
    lr=lr,
    decay=decay,
    dtype='float32',dtypeL='uint8',
    wType='slice',
    loss='categorical_crossentropy',
    metric='accuracy',
    model='unet')


logFile.write('\n'+'-'*30+'\n')
logFile.write(unetSS.__str__()) 
logFile.write('\n'+'-'*30+'\n')
logFile.write(datagenPrams)
logFile.write('\n'+'-'*30+'\n')
logFile.close()


if __name__ == '__main__':
    unetSS.pred_folder=pred_folder
    unetSS.save_folder = os.path.join(code_dir,resultsFolder,oLabel,pred_folder)
    unetSS.weight_folder=os.path.join(code_dir,'weights_folder')
    
    if predict:
        if IMAGE_INPUT == '2D':
            # 2D pre-trained by authors
            weightFile=(os.path.join(unetSS.weight_folder, WEIGHTS_2D_ORIGINAL))
            unetSS.Predict(weightFile)
        if IMAGE_INPUT == '3D':
            # 3D pre-trained by authors
            weightFile=(os.path.join(unetSS.weight_folder, WEIGHTS_3D_ORIGINAL))
            unetSS.Predict3D(weightFile)
        else:
            print('please set IMAGE_INPUT')
            sys.exit()

    elif train:
        if IMAGE_INPUT == '2D':
            # 2D pre-trained by authors
            weightFile=(os.path.join(unetSS.weight_folder, WEIGHTS_2D_ORIGINAL))
            unetSS.train(train_from_scratch=TRAIN_FROM_SCRATCH, pretrained_weights = weightFile)
        
        if IMAGE_INPUT == '3D':
            # 3D pre-trained by authors
            weightFile=(os.path.join(unetSS.weight_folder, WEIGHTS_3D_ORIGINAL))
            unetSS.train3D(train_from_scratch=TRAIN_FROM_SCRATCH, pretrained_weights = weightFile)

    else:
        print('please set a task flag: train or predict')
