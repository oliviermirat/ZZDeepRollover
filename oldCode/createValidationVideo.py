from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
import sys
import json
import pandas as pd

def createValidationVideo(videoName, path, rolloversMedFiltAllWells, resultsPercentages):

  ### Loading the images and applying the classifier on them

  videoPath = path + videoName + '/results_' + videoName + '.txt'
  
  resizeScale = 3
  frame_width = 30
  frame_height = 30
  ext1 = 20
  ext2 = 20
  
  font                   = cv2.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (1,10)
  fontScale              = 0.4
  fontColor              = (255,255,255)
  lineType               = 1
  y_offset               = 0
  x_offset               = 0
  
  rolloverFrameFile = path + videoName + '/rolloverManualClassification.json'
  exists = os.path.isfile(rolloverFrameFile)
  if exists:
    videoPath2 = path+videoName+'/'+videoName+'_'+str(1)+'/'+videoName+'_'+str(1)+'.avi'
    cap = cv2.VideoCapture(videoPath2)
    videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    # Loading "true" maybes
    trueMaybeAllWells = []
    for well in rolloverFrame:
      trueMaybe = np.zeros((videoLength))
      for boundaries in rolloverFrame[well]['inBetween']:
        left  = boundaries[0]
        right = boundaries[1]
        for i in range(left,right+1):
          trueMaybe[i] = 1
      trueMaybeAllWells.append(trueMaybe)
    trueMaybeAllWells = np.array(trueMaybeAllWells)
  else:
    trueRolloverAllWells = []
  
  out2 = cv2.VideoWriter(path + videoName + '/rolloverValidationAllFrames.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (resizeScale*frame_width + ext1, resizeScale*frame_height + ext2))
  
  if len(rolloversMedFiltAllWells):
    out = cv2.VideoWriter(path + videoName + '/validationOnlyFramesDetectedAsRollover.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (resizeScale*frame_width + ext1, resizeScale*frame_height + ext2))
    out3 = cv2.VideoWriter(path + videoName + '/validationOnlyFramesDetectedAsNormal.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (resizeScale*frame_width + ext1, resizeScale*frame_height + ext2))
    out4 = cv2.VideoWriter(path + videoName + '/validationOnlyErrors.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (resizeScale*frame_width + ext1, resizeScale*frame_height + ext2))

  if (os.path.isfile(videoPath)):
    
    # Applying rollover classifier to each frame and saving the results in a txt file
    file = open(videoPath, 'r')
    j = json.loads(file.read())
    wellPoissMouv = j['wellPoissMouv']
    wellPositions = j['wellPositions']
    nbWell = len(wellPositions)

    # going through each well in super structure
    for i in range(0,nbWell):
      xwell = wellPositions[i]['topLeftX']
      ywell = wellPositions[i]['topLeftY']
      if xwell < 0:
        xwell = 0
      if ywell < 0:
        ywell = 0
      videoPath2 = path+videoName+'/'+videoName+'_'+str(i+1)+'/'+videoName+'_'+str(i+1)+'.avi'
      if (len(wellPoissMouv[i])):
        if (len(wellPoissMouv[i][0])):
          cap = cv2.VideoCapture(videoPath2)
          nbMouv = len(wellPoissMouv[i][0])
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
                  frame2 = cv2.resize(frame, (resizeScale*frame_width, resizeScale*frame_height))
                  
                  frame3 = np.zeros((frame2.shape[0] + ext1, frame2.shape[1] + ext2, 3), np.uint8)
                  frame3[0:frame2.shape[0],0:frame2.shape[1]] = frame2
                  
                  if len(rolloversMedFiltAllWells):
                    if rolloversMedFiltAllWells[i][k]:
                      cv2.circle(frame3,(10,10),10,(0,0,255),-1)
                    if len(trueRolloverAllWells):
                      if trueRolloverAllWells[i][k]:
                        cv2.circle(frame3,(10,29),10,(0,255,0),-1)
                    if len(trueMaybeAllWells):
                      if trueMaybeAllWells[i][k]:
                        cv2.circle(frame3,(10,25),10,(0,165,255),-1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame3, str((int(resultsPercentages[i][k]*100)))+"%", (10, resizeScale*frame_height+ext2-7), font, 0.5, (0, 0, 255), 2)
                    
                    cv2.putText(frame3, str(i+1)+";"+str(k), (resizeScale*frame_width-40, resizeScale*frame_height+ext2-7), font, 0.4, (0, 0, 255), 1)

                    out2.write(frame3)
                    if rolloversMedFiltAllWells[i][k]:
                      out.write(frame3)
                    else:
                      out3.write(frame3)
                      
                    if trueMaybeAllWells[i][k] == 0:
                      if rolloversMedFiltAllWells[i][k] != trueRolloverAllWells[i][k]:
                        out4.write(frame3)
                  
                  else:
                    
                    if len(trueRolloverAllWells):
                      if trueRolloverAllWells[i][k]:
                        cv2.circle(frame3,(10,12),10,(0,0,255),-1)
                      if trueMaybeAllWells[i][k]:
                        cv2.circle(frame3,(10,15),10,(0,165,255),-1)
                        
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame3, str(i+1)+";"+str(k), (resizeScale*frame_width-40, resizeScale*frame_height+ext2-7), font, 0.4, (0, 0, 255), 1)
                    out2.write(frame3)
                  
                else:
                  break
                k = k + 1
  
  out2.release()
  if len(rolloversMedFiltAllWells):
    out.release()
    out3.release()
    out4.release()


if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

  videoName = sys.argv[1]
  path = sys.argv[2]
  
  createValidationVideo(videoName,path,[],[])
  