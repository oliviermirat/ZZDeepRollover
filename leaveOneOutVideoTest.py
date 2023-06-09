from createInitialImagesNewZZ import createInitialImagesNewZZ
from createTrainOrTestDataset import createTrainOrTestDataset
from learnModel import learnModel
from detectRolloverFramesWithNewZZversion import detectRolloverFramesWithNewZZversion
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd
import os
from cleanFolders import cleanDataset, cleanModel

# Size of image on which DL algorithm will be applied will be 2*(recenterImageWindow-int(2*recenterImageWindow/6))
recenterImageWindow          = 34 # 24
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 100
# Number of rotations applied on image for training
numberOfRotationsDataAugmentationTraining=12
# Number of epoch for DL training
epochsNbTraining=10 #15
# Window of median rolling mean applied on rollover detected
medianRollingMean=5

videos = ['20190727-13-2-1', '20190427-1-2-5', '20190503-2-2-3', '20190727-6-2-4', '20190503-1-2-5', '20190427-3-2-2', '20190427-1-2-2', '20190427-4-2-1', '20190503-4-2-7', '20190727-2-2-1', '20190727-9-2-7', '20190427-1-2-1', '20190427-5-2-1', '20190727-10-2-7', '20190727-3-2-2', '20190427-2-2-6', '20190727-11-2-7', '20190727-4-2-6', '20190427-2-2-8', '20190503-2-2-1', '20190727-1-2-3', '20190727-5-2-7']

pathToRawVideos = 'ZZoutputNew'

###

results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])

for video in videos:
  createInitialImagesNewZZ(video, 'rolloverManualClassification.json', pathToRawVideos + '/', imagesToClassifyHalfDiameter, 0)

for idx, video in enumerate(videos):

  trainingVid = videos.copy()
  testingVid = trainingVid.pop(idx)
  
  print("\nTestingVid:", testingVid, "\n")
  
  cleanDataset(os.path.join('trainingDataset', 'normal'), 'png')
  cleanDataset(os.path.join('trainingDataset', 'rollover'), 'png')
  cleanDataset(os.path.join('testDataset', 'normal'), 'png')
  cleanDataset(os.path.join('testDataset', 'rollover'), 'png')
  cleanModel('model')

  # Creating learning dataset
  for vidTrain in trainingVid:
    createTrainOrTestDataset('initialImages/',vidTrain,'trainingDataset',numberOfRotationsDataAugmentationTraining,recenterImageWindow)
  
  # Transfert learning
  learnModel('trainingDataset', epochsNbTraining, 'model', False)

  # Testing the model on the entire video
  validationVideo = 1
  pathToRawVideo  = os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi') if os.path.exists(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi')) else os.path.join(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid), testingVid + '.seq')
  [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo] = detectRolloverFramesWithNewZZversion(testingVid, pathToRawVideos + '/', medianRollingMean, recenterImageWindow, 1, validationVideo, pathToRawVideo, imagesToClassifyHalfDiameter)
  
  if totalTrueNormal:
    normalClassedAsRolloPercent  = (normalClassedAsRollo/totalTrueNormal) * 100
  else:
    normalClassedAsRolloPercent  = 0
  if totalTrueRollo:
    rolloClassedAsRolloNbPercent = (rolloClassedAsRollo/totalTrueRollo) * 100
  else:
    rolloClassedAsRolloNbPercent = 100
  
  results.loc[testingVid] = [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo, normalClassedAsRolloPercent, rolloClassedAsRolloNbPercent]

print("Final Results")
df = results[['normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent']]
print(df)
print("")
print("normalClassedAsRolloPercent average:", df["normalClassedAsRolloPercent"].mean(), "%")
print("rolloClassedAsRolloNbPercent average:", df["rolloClassedAsRolloNbPercent"].mean(), "%")
