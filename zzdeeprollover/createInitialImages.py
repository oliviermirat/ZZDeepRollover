import zzdeeprollover.zzVideoReading as zzVideoReading
import json
import os
import cv2
import shutil
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import csv
import sys

def createInitialImages(videoName, rolloverFrameFile, pathToZZoutput, imagesToClassifyHalfDiameter, initialImagesFolder, backgroundRemoval=0):
  
  if not(os.path.isdir(initialImagesFolder)):
    os.mkdir(initialImagesFolder)
  
  if (os.path.isdir(initialImagesFolder+'/'+videoName)):
    shutil.rmtree(initialImagesFolder+'/'+videoName)
  os.mkdir(initialImagesFolder+'/'+videoName)
  os.mkdir(initialImagesFolder+'/'+videoName+'/rollover')
  os.mkdir(initialImagesFolder+'/'+videoName+'/normal')

  rolloverFrameFile = pathToZZoutput + videoName + '/' + rolloverFrameFile

  csvFileName = videoName

  videoPath = pathToZZoutput + videoName + '/results_' + videoName + '.txt'
  
  if backgroundRemoval:
    backgroundPath = pathToZZoutput + videoName + '/background.png'
    background     = cv2.imread(backgroundPath)
    background     = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
  
  if (os.path.isfile(videoPath)):

    fileRollover = open(rolloverFrameFile, 'r')
    rolloverFrame = json.loads(fileRollover.read())

    # opening super structure
    file = open(videoPath, 'r')
    jsonFile = json.loads(file.read())
    wellPoissMouv = jsonFile['wellPoissMouv']
    wellPositions = jsonFile['wellPositions']
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
      
      if 'pathToOriginalVideo' in jsonFile:
        videoPath2 = jsonFile['pathToOriginalVideo']
        if not(os.path.exists(videoPath2)):
          if 'alternativePathToOriginalVideo' in jsonFile:
            videoPath2 = jsonFile['alternativePathToOriginalVideo']
            if not(os.path.exists(videoPath2)):
              raise("fix video path issue in result file")
          else:
            raise("fix video path issue in result file")
      else:
        if 'alternativePathToOriginalVideo' in jsonFile:
          videoPath2 = jsonFile['alternativePathToOriginalVideo']
          if not(os.path.exists(videoPath2)):
            raise("fix video path issue in result file")
        else:
          raise("fix video path issue in result file")
      
      if videoPath2[:30] == '\\\\l2export\\iss02.wyart\\rawdata':
        videoPath2 = videoPath2.replace('\\', '/')
        videoPath2.replace('//l2export/iss02.wyart/', '/network/lustre/iss02/wyart/')
      
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
                
                if yStart == 0:
                  yEnd = 2 * imagesToClassifyHalfDiameter
                if xStart == 0:
                  xEnd = 2 * imagesToClassifyHalfDiameter
                if yEnd == len(frame):
                  yStart = len(frame) - 2 * imagesToClassifyHalfDiameter
                if xEnd == len(frame[0]):
                  xStart = len(frame[0]) - 2 * imagesToClassifyHalfDiameter
                
                frame = frame[yStart:yEnd, xStart:xEnd]
                
                if ret == True:
                  # Saving image
                  if not(k in inBetween):
                    if k in rollover:
                      cv2.imwrite(initialImagesFolder + '/' + videoName + '/rollover/img' + str(m) + '.png', frame)
                    else:
                      cv2.imwrite(initialImagesFolder + '/' + videoName + '/normal/img' + str(m) + '.png', frame)
                  m = m + 1
                  
                else: 
                  break
                k = k + 1
    
    cap.release()
