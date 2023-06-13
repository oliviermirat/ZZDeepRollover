from zzdeeprollover.cleanFolders import refreshTrainingDataset, cleanModel, removeIpynbCheckpointsFromTrainingDataset
from zzdeeprollover.createInitialImages import createInitialImages
from zzdeeprollover.createTrainOrTestDataset import createTrainOrTestDataset
from zzdeeprollover.learnModel import learnModel
from zzdeeprollover.detectRolloverFrames import detectRolloverFrames
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd
import os

generateInitialImages = True
generateTrainingDataset = True
localComputer = True

initialImagesFolder = 'initialImages' if localComputer else 'drive/MyDrive/initialImages'

# Size of image on which DL algorithm will be applied
resizeCropDimension          = 34
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 50
# Number of epoch for DL training
epochsNbTraining = 1 if localComputer else 10
# Window of median rolling mean applied on rollover detected
medianRollingMean = 5

file_path = 'listOfVideosToTakeIntoAccount.txt'
with open(file_path, 'r') as file:
  lines = file.readlines()
  videos = [line.strip() for line in lines]
print("Videos taken into account:", videos)

pathToRawVideos = 'ZZoutputNew' if localComputer else 'drive/MyDrive/ZZoutputNew'

###

if __name__ == '__main__':

  results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])

  if generateInitialImages:
    for video in videos:
      createInitialImages(video, 'rolloverManualClassification.json', pathToRawVideos + '/', imagesToClassifyHalfDiameter, initialImagesFolder)

  for idx, video in enumerate(videos):

    trainingVid = videos.copy()
    testingVid  = trainingVid.pop(idx)
    
    print("\nTestingVid:", testingVid, "\n")
    
    if generateTrainingDataset:
      refreshTrainingDataset()
    cleanModel('model')

    # Creating learning dataset
    if generateTrainingDataset:
      for vidTrain in trainingVid:
        createTrainOrTestDataset(initialImagesFolder + '/', vidTrain, os.path.join('trainingDataset','train'), int(resizeCropDimension/2))
      createTrainOrTestDataset(initialImagesFolder + '/', testingVid, os.path.join('trainingDataset','val'), int(resizeCropDimension/2))
      print("Training and testing dataset created")
    
    # Transfert learning
    removeIpynbCheckpointsFromTrainingDataset()
    learnModel(epochsNbTraining, 'model', resizeCropDimension)
    
    # Testing the model on the entire video
    validationVideo = 1
    pathToRawVideo  = os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi') if os.path.exists(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi')) else os.path.join(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid), testingVid + '.seq')
    [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo] = detectRolloverFrames(testingVid, pathToRawVideos + '/', medianRollingMean, resizeCropDimension, 1, validationVideo, pathToRawVideo, imagesToClassifyHalfDiameter, os.path.join('model', 'model.pth'))
    
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
