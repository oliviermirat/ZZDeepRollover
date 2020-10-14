import json
import os
import cv2
import shutil
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import csv
import sys

def createInitialImages(videoName, rolloverFrameFile, path):

  if (os.path.isdir('initialImages/'+videoName)):
    shutil.rmtree('initialImages/'+videoName)
  os.mkdir('initialImages/'+videoName)
  os.mkdir('initialImages/'+videoName+'/rollover')
  os.mkdir('initialImages/'+videoName+'/normal')

  rolloverFrameFile = path + videoName + '/' + rolloverFrameFile

  csvFileName = videoName

  videoPath = path + videoName + '/results_' + videoName + '.txt'

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
      
      videoPath2 = path+videoName+'/'+videoName+'_'+str(i+1)+'/'+videoName+'_'+str(i+1)+'.avi'
      vidPosPath = path+videoName+'/'+videoName+'_'+str(i+1)+'/videoCreatedXYborder.txt'
      if (len(wellPoissMouv[i])):
        if (len(wellPoissMouv[i][0])):
          cap = cv2.VideoCapture(videoPath2)
          nbMouv = len(wellPoissMouv[i][0])
          
          dfVidPosPath = pd.read_csv(vidPosPath)
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
