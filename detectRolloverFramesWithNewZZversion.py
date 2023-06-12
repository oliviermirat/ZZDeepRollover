import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from tempfile import TemporaryDirectory

from PIL import Image
to_pil = transforms.ToPILImage()

# from __future__ import absolute_import, division, print_function
import zzVideoReading as zzVideoReading
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
import sys
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from imageTransformFunctions import recenterImageOnEyes
from createValidationVideoWithNewZZversion import createValidationVideoWithNewZZversion

showImagesUsedForTraining = False

def runModelOnFrames(frames, model, resizeCropDimension):
  
  frames = np.array(frames)
      
  data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resizeCropDimension),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
  ])
  
  if showImagesUsedForTraining:
    if False:
      for inputImg in [frames[i] for i in range(0, len(frames))]:
        # pil_image = to_pil((inputImg*255).astype(np.uint8))
        pil_image = to_pil(inputImg)
        pil_image.show()
    else:
      for inputImg in torch.stack([data_transform(frames[i].astype(np.uint8)) for i in range(0, len(frames))]):
        pil_image = to_pil(inputImg)
        pil_image.show()
  
  outputs = model(torch.stack([data_transform(frames[i].astype(np.uint8)) for i in range(0, len(frames))]))
  _, result = torch.max(outputs, 1)
  
  binary = result
  probabilities = outputs.detach().numpy()[:, 0]
      
  return [binary, probabilities]


def detectRolloverFramesWithNewZZversion(videoName, path, medianRollingMean, resizeCropDimension, comparePredictedResultsToManual, validationVideo, pathToInitialVideo, imagesToClassifyHalfDiameter, backgroundRemoval=0):
  
  if (medianRollingMean % 2 == 0):
    sys.exit("medianRollingMean must be an odd number")

  ### Loading the classifier
  model = models.resnet50()
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 2)
  model.load_state_dict(torch.load(os.path.join('model', 'model.pth')))
  model.eval()
  
  ### Background extraction
  if backgroundRemoval:
    backgroundPath = path + videoName + '/background.png'
    background     = cv2.imread(backgroundPath)
    background     = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

  ### Loading the images and applying the classifier on them

  videoPath = os.path.join(os.path.join(path, videoName), 'results_' + videoName + '.txt')

  if (os.path.isfile(videoPath)):
    
    # Applying rollover classifier to each frame and saving the results in a txt file
    
    file = open(videoPath, 'r')
    j = json.loads(file.read())
    wellPoissMouv = j['wellPoissMouv']
    wellPositions = j['wellPositions']
    nbWell = len(wellPositions)
    rolloversAllWells = []
    rolloversMedFiltAllWells = []
    rolloverPercentageAllWells = []
    # going through each well in super structure
    for i in range(0,nbWell):
      print("Running rollover detection on video", videoName, "for well:", i)
      xwell = wellPositions[i]['topLeftX']
      ywell = wellPositions[i]['topLeftY']
      if xwell < 0:
        xwell = 0
      if ywell < 0:
        ywell = 0
      videoPath2 = pathToInitialVideo
      frames = []
      framesNumber = []
      cap = zzVideoReading.VideoCapture(videoPath2)
      videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      rollovers = np.zeros((videoLength))
      rolloverPercentage = np.zeros((videoLength))
      allBinary        = np.array([])
      allProbabilities = np.array([])
      if (len(wellPoissMouv[i])):
        if (len(wellPoissMouv[i][0])):
          nbMouv = len(wellPoissMouv[i][0])
          # going through each movement for the well
          for j in range(0,nbMouv):
            if (len(wellPoissMouv[i][0][j])):
              item = wellPoissMouv[i][0][j]
              BoutStart = item['BoutStart']
              BoutEnd   = item['BoutEnd']
              k = BoutStart
              cap.set(cv2.CAP_PROP_POS_FRAMES,BoutStart)
              while (k <= BoutEnd):
                ret, frame = cap.read()
                
                if ret == True:
                
                  if backgroundRemoval:
                    minPixelDiffForBackExtract = 15
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    putToWhite = ( frame.astype('int32') >= (background.astype('int32') - minPixelDiffForBackExtract) )
                    frame[putToWhite] = int(np.mean(np.mean(frame)))
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)                
                  
                  yStart = int(ywell+item['HeadY'][k-BoutStart]-int(imagesToClassifyHalfDiameter))
                  yEnd   = int(ywell+item['HeadY'][k-BoutStart]+int(imagesToClassifyHalfDiameter))
                  xStart = int(xwell+item['HeadX'][k-BoutStart]-int(imagesToClassifyHalfDiameter))
                  xEnd   = int(xwell+item['HeadX'][k-BoutStart]+int(imagesToClassifyHalfDiameter))
                  
                  if xStart < 0:
                    xStart = 0
                  if yStart < 0:
                    yStart = 0
                  if xEnd >= len(frame[0]):
                    xEnd = len(frame[0])
                  if yEnd >= len(frame):
                    yEnd = len(frame)
                  
                  if yStart == 0:
                    yEnd = 2 * imagesToClassifyHalfDiameter
                  if xStart == 0:
                    xEnd = 2 * imagesToClassifyHalfDiameter
                  if yEnd == len(frame):
                    yStart = len(frame) - 2 * imagesToClassifyHalfDiameter
                  if xEnd == len(frame[0]):
                    xStart = len(frame[0]) - 2 * imagesToClassifyHalfDiameter
                  
                  frame = frame[yStart:yEnd, xStart:xEnd]
                  frame = recenterImageOnEyes(frame, int(resizeCropDimension/2))
                  frames.append(frame)
                  framesNumber.append(k)
                  
                  if len(frames) >= 10:
                    [binary, probabilities] = runModelOnFrames(frames, model, resizeCropDimension)
                    allBinary               = np.append(allBinary, binary)
                    allProbabilities        = np.append(allProbabilities, probabilities)
                    frames = []
                
                else:
                  break
                k = k + 1
        
          if len(frames) > 0:
            [binary, probabilities] = runModelOnFrames(frames, model, resizeCropDimension)
            allBinary               = np.append(allBinary, binary)
            allProbabilities        = np.append(allProbabilities, probabilities)
          
          if len(framesNumber) > 0:
            rollovers[framesNumber] = allBinary
            rolloverPercentage[framesNumber] = allProbabilities
          
      rolloversMedFiltSeries = (pd.Series(rollovers)).rolling(medianRollingMean).median()
      for i in range(0, medianRollingMean):
        rolloversMedFiltSeries[i] = 0
      rolloversMedFiltSeries = np.roll(rolloversMedFiltSeries,int(-((medianRollingMean-1)/2)))
      
      rolloversAllWells.append(rollovers)
      rolloversMedFiltAllWells.append(rolloversMedFiltSeries)
      rolloverPercentageAllWells.append(rolloverPercentage)
    
    rolloversAllWells = np.array(rolloversAllWells)
    rolloversMedFiltAllWells = np.array(rolloversMedFiltAllWells)
    cap.release()
    np.savetxt(os.path.join(os.path.join(path, videoName), 'rolloverClassified.txt'), rolloversMedFiltAllWells, fmt='%d')
    np.savetxt(os.path.join(os.path.join(path, videoName), 'rolloverPercentages.txt'), rolloverPercentageAllWells, fmt='%f')
    
    if validationVideo:
      createValidationVideoWithNewZZversion(videoName, path, rolloversMedFiltAllWells, rolloverPercentageAllWells, pathToInitialVideo, int(resizeCropDimension/2))
    
    if comparePredictedResultsToManual:
    
      # Printing rollovers frames before the rolling median filter
      
      rollForAllWells = []
      for rollover in rolloversAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWells.append(rollForCurWell)
      
      print("Before rolling medfilter:")
      print(rollForAllWells)
      
      # Printing the rollover after the rolling median filter

      rollForAllWellsMed = []
      for rollover in rolloversMedFiltAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWellsMed.append(rollForCurWell)
      
      print("\nAfter rolling medfilter:")
      print(rollForAllWellsMed)
      
      # Printing manual classifications of rollovers
      
      print("\nManual classification:")
      rolloverFrameFile = os.path.join(os.path.join(path, videoName), 'rolloverManualClassification.json')
      fileRollover = open(rolloverFrameFile, 'r')
      rolloverFrame = json.loads(fileRollover.read())
      
      trueRolloverAllWells = []
      for well in rolloverFrame:
        trueRollovers = np.zeros((videoLength))
        for boundaries in rolloverFrame[well]['rollover']:
          left  = boundaries[0]
          right = boundaries[1]
          for i in range(left,right+1):
            trueRollovers[i] = 1
        trueRolloverAllWells.append(trueRollovers)
      trueRolloverAllWells = np.array(trueRolloverAllWells)
      
      wellNumberAllWells = []
      for idx, well in enumerate(rolloverFrame):
        wellNumber = np.zeros((videoLength))
        # for boundaries in rolloverFrame[well]['rollover']:
          # left  = boundaries[0]
          # right = boundaries[1]
          # for i in range(left,right+1):
            # wellNumber[i] = idx+1
        for idx2, aaa in enumerate(wellNumber):
          wellNumber[idx2] = idx+1
        wellNumberAllWells.append(wellNumber)
      wellNumberAllWells = np.array(wellNumberAllWells)
      
      boutNumberAllWells = []
      for idx, well in enumerate(rolloverFrame):
        boutNumber = np.zeros((videoLength))
        # for boundaries in rolloverFrame[well]['rollover']:
          # left  = boundaries[0]
          # right = boundaries[1]
          # for i in range(left,right+1):
            # boutNumber[i] = i
        for idx2, aaa in enumerate(boutNumber):
          boutNumber[idx2] = idx2
        boutNumberAllWells.append(boutNumber)
      boutNumberAllWells = np.array(boutNumberAllWells)
      
      rollForAllWellsTrue = []
      for rollover in trueRolloverAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWellsTrue.append(rollForCurWell)
      
      print(rollForAllWellsTrue)
      print("")
      print(rolloverFrame)
      
      # Getting the frames in-between "rollovers" and "normal"
      
      inBetweensAllWells = []
      for well in rolloverFrame:
        inBetweens = np.zeros((videoLength))
        for boundaries in rolloverFrame[well]['inBetween']:
          left  = boundaries[0]
          right = boundaries[1]
          for i in range(left,right+1):
            inBetweens[i] = i
        inBetweens = inBetweens.tolist()
        inBetweensAllWells.append(inBetweens)
        
      # Removing the "in-between" frames from the before medfilt rollovers, after medfilt rollovers, and the trues rollovers
      
      rolloversMedFiltAllWellsInBetweenRemoved = []
      rolloversAllWellsInBetweenRemoved        = []
      trueRolloverAllWellsInBetweenRemoved     = []
      wellNumberAllWellsInBetweenRemoved       = []
      boutNumberAllWellsInBetweenRemoved       = []
      # pdb.set_trace()
      for idx, inBetweens in enumerate(inBetweensAllWells):
        rolloversMedFiltAllWellsInBetweenRemoved.extend(np.delete(rolloversMedFiltAllWells[idx], np.array(inBetweens).astype(int).tolist()).tolist())
        rolloversAllWellsInBetweenRemoved.extend(np.delete(rolloversAllWells[idx], np.array(inBetweens).astype(int).tolist()).tolist())
        trueRolloverAllWellsInBetweenRemoved.extend(np.delete(trueRolloverAllWells[idx], np.array(inBetweens).astype(int).tolist()).tolist())
        wellNumberAllWellsInBetweenRemoved.extend(np.delete(wellNumberAllWells[idx], np.array(inBetweens).astype(int).tolist()).tolist())
        boutNumberAllWellsInBetweenRemoved.extend(np.delete(boutNumberAllWells[idx], np.array(inBetweens).astype(int).tolist()).tolist())
      rolloversMedFiltAllWellsInBetweenRemoved = np.array(rolloversMedFiltAllWellsInBetweenRemoved)
      rolloversAllWellsInBetweenRemoved        = np.array(rolloversAllWellsInBetweenRemoved)
      trueRolloverAllWellsInBetweenRemoved     = np.array(trueRolloverAllWellsInBetweenRemoved)
      wellNumberAllWellsInBetweenRemoved       = np.array(wellNumberAllWellsInBetweenRemoved)
      boutNumberAllWellsInBetweenRemoved       = np.array(boutNumberAllWellsInBetweenRemoved)
      
      # Using scikit learn to print classification statistics
      
      print(classification_report(trueRolloverAllWellsInBetweenRemoved, rolloversAllWellsInBetweenRemoved))
      print(classification_report(trueRolloverAllWellsInBetweenRemoved, rolloversMedFiltAllWellsInBetweenRemoved))
      
      print(precision_recall_fscore_support(trueRolloverAllWellsInBetweenRemoved, rolloversAllWellsInBetweenRemoved))
      print(precision_recall_fscore_support(trueRolloverAllWellsInBetweenRemoved, rolloversMedFiltAllWellsInBetweenRemoved))
      
      # Printing "normal classified as rollover" percentage and "rollover classified as rollover" percentage and returning those two values
      
      normalClassedAsRollo = 0
      rolloClassedAsRollo = 0
      totalTrueNormal = 0
      totalTrueRollo  = 0
      for idx,true in enumerate(trueRolloverAllWellsInBetweenRemoved):
        pred = rolloversMedFiltAllWellsInBetweenRemoved[idx]
        if true == 0:
          totalTrueNormal = totalTrueNormal + 1
          if pred == 1:
            normalClassedAsRollo = normalClassedAsRollo + 1
            print("False positive: well:",wellNumberAllWellsInBetweenRemoved[idx]," ; bout:",boutNumberAllWellsInBetweenRemoved[idx])
        if true == 1:
          totalTrueRollo = totalTrueRollo + 1
          if pred == 1:
            rolloClassedAsRollo = rolloClassedAsRollo + 1
          else:
            print("False negative: well:",wellNumberAllWellsInBetweenRemoved[idx]," ; bout:",boutNumberAllWellsInBetweenRemoved[idx])
      
      if totalTrueNormal:
        print("normal classified as rollover:",normalClassedAsRollo,"out of",totalTrueNormal,"; so:",(normalClassedAsRollo/totalTrueNormal)*100,"%")
      else:
        print("no true normal in this dataset")
      if totalTrueRollo:
        print("rollover classified as rollover:",rolloClassedAsRollo,"out of",totalTrueRollo,"; so:",(rolloClassedAsRollo/totalTrueRollo)*100,"%")
      else:
        print("no true rollovers in this dataset")
        
      return [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo]
