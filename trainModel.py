from createInitialImagesNewZZ import createInitialImagesNewZZ
from createTrainOrTestDataset import createTrainOrTestDataset
from learnModel import learnModel
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd
import os
from cleanFolders import refreshTrainingDataset, cleanModel, removeIpynbCheckpointsFromTrainingDataset

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

pathToRawVideos = 'ZZoutput' if localComputer else 'drive/MyDrive/ZZoutputNew'

###

if __name__ == '__main__':

  results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])
  
  if generateInitialImages:
    for video in videos:
      createInitialImagesNewZZ(video, 'rolloverManualClassification.json', pathToRawVideos + '/', imagesToClassifyHalfDiameter, initialImagesFolder)


  trainingVid = videos.copy()

  if generateTrainingDataset:
    refreshTrainingDataset()
  cleanModel('model')

  # Creating learning dataset
  for vidTrain in trainingVid:
    createTrainOrTestDataset(initialImagesFolder + '/', vidTrain, os.path.join('trainingDataset','train'), int(resizeCropDimension/2))

  # Transfert learning
  removeIpynbCheckpointsFromTrainingDataset()
  learnModel(epochsNbTraining, 'model', resizeCropDimension)
