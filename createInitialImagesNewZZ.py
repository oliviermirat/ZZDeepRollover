import json
import os
import cv2
import shutil
import videoFormatConversion.zzVideoReading as zzVideoReading
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import csv
import sys

def createInitialImagesNewZZ(videoName, rolloverFrameFile, path, imagesToClassifyHalfDiameter, backgroundRemoval=0):
  
  recenterImageWindow = 0
  
  if (os.path.isdir('initialImages/'+videoName)):
    shutil.rmtree('initialImages/'+videoName)
  os.mkdir('initialImages/'+videoName)
  os.mkdir('initialImages/'+videoName+'/rollover')
  os.mkdir('initialImages/'+videoName+'/normal')

  rolloverFrameFile = path + videoName + '/' + rolloverFrameFile

  csvFileName = videoName

  videoPath = path + videoName + '/results_' + videoName + '.txt'
  
  if backgroundRemoval:
    backgroundPath = path + videoName + '/background.png'
    background     = cv2.imread(backgroundPath)
    background     = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
  
  if (os.path.isfile(videoPath)):

    fileRollover = open(rolloverFrameFile, 'r')
    rolloverFrame = json.loads(fileRollover.read())

    # opening super structure
    file = open(videoPath, 'r')
    j = json.loads(file.read())
    wellPoissMouv = j['wellPoissMouv']
    wellPositions = j['wellPositions']
    nbWell = len(wellPoissMouv)
    m = 0
    
    # going through each well in super structure
    for i in range(0,nbWell):
      rolloverRanges = rolloverFrame[str(i+1)]["rollover"]
      inBetweenRanges = rolloverFrame[str(i+1)]["inBetween"]
      rollover = []
      inBetween = []
      for rangeBoundary in rolloverRanges:
        for value in range(rangeBoundary[0], rangeBoundary[1]+1):
          rollover.append(value)
      for rangeBoundary in inBetweenRanges:
        for value in range(rangeBoundary[0], rangeBoundary[1]+1):
          inBetween.append(value)
          
      print(rollover)
      print(inBetween)
      
      xwell = wellPositions[i]['topLeftX']
      ywell = wellPositions[i]['topLeftY']
      if xwell < 0:
        xwell = 0
      if ywell < 0:
        ywell = 0
      
      if os.path.exists(path+videoName+'/'+videoName+'.avi'):
        videoPath2 = path+videoName+'/'+videoName+'.avi'
      else:
        videoPath2 = path+videoName+'/'+videoName+'/'+videoName+'.seq'
      if (len(wellPoissMouv[i])):
        if (len(wellPoissMouv[i][0])):
          cap = zzVideoReading.VideoCapture(videoPath2)
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
                
                if backgroundRemoval:
                  minPixelDiffForBackExtract = 15
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  putToWhite = ( frame.astype('int32') >= (background.astype('int32') - minPixelDiffForBackExtract) )
                  frame[putToWhite] = int(np.mean(np.mean(frame)))
                  frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                yStart = int(ywell+item['HeadY'][k-BoutStart]-imagesToClassifyHalfDiameter)
                yEnd   = int(ywell+item['HeadY'][k-BoutStart]+imagesToClassifyHalfDiameter)
                xStart = int(xwell+item['HeadX'][k-BoutStart]-imagesToClassifyHalfDiameter)
                xEnd   = int(xwell+item['HeadX'][k-BoutStart]+imagesToClassifyHalfDiameter)
                if xStart < 0:
                  xStart = 0
                if yStart < 0:
                  yStart = 0
                if xEnd >= len(frame[0]):
                  xEnd = len(frame[0]) - 1
                if yEnd >= len(frame):
                  yEnd = len(frame) - 1
                frame = frame[yStart:yEnd, xStart:xEnd]
                if ret == True:
                  if recenterImageWindow:
                    frame = recenterImageOnEyes(frame,recenterImageWindow)
                  rows = len(frame)
                  cols = len(frame[0])
                  scaleD = int(cols/6)
                  frame = frame[scaleD:(rows-scaleD), scaleD:(rows-scaleD)]
                  frame = cv2.resize(frame,(224,224))
                  # frame = np.array(frame, dtype=np.float32) / 255.0
                
                if ret == True:
                  # print(["frame:",k," ; corresponds to m:",m])
                  # Saving image
                  if not(k in inBetween):
                    if k in rollover:
                      cv2.imwrite('initialImages/' + videoName + '/rollover/img' + str(m) + '.png', frame)
                    else:
                      cv2.imwrite('initialImages/' + videoName + '/normal/img' + str(m) + '.png', frame)
                  m = m + 1
                  
                else: 
                  break
                k = k + 1
    
    cap.release()


if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
  
  videoName = sys.argv[1]
  rolloverFrameFile = sys.argv[2]
  path = sys.argv[3]
  createInitialImages(videoName, rolloverFrameFile, path)
