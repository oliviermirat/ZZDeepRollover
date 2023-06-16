from zzdeeprollover.cleanFolders import refreshTrainingDataset, cleanModel, removeIpynbCheckpointsFromTrainingDataset
from zzdeeprollover.createInitialImages import createInitialImages
from zzdeeprollover.createTrainOrTestDataset import createTrainOrTestDataset
from zzdeeprollover.learnModel import learnModel
from zzdeeprollover.addAlternativePathToOriginalVideo import addAlternativePathToOriginalVideo
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd
import random
import os

localComputer = True

generateInitialImages = True
generateTrainingDataset = True

# Size of image on which DL algorithm will be applied
resizeCropDimension          = 34
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 50
# Number of epoch for DL training
epochsNbTraining = 1 if localComputer else 10
# Window of median rolling mean applied on rollover detected
medianRollingMean = 5

if True:
  learningParameters = {}
else:
  learningParameters = {'maxCrop': 1, 'brightness_limit': 0.05, 'contrast_limit': 0.05, 'invert_probability': 0, 'sharpness_probability': 0}

showImagesUsedForTraining = False

initialImagesFolder = 'initialImages' if localComputer else 'drive/MyDrive/initialImages'

pathToZZoutput = 'ZZoutputNew' if localComputer else 'drive/MyDrive/ZZoutputNew'

file_path = 'listOfVideosToTakeIntoAccount.txt'
with open(file_path, 'r') as file:
  lines = file.readlines()
  videos = [line.strip() for line in lines]

# If launched on a computer other than the one used to launch the tracking, the paths to the original raw videos saved in the result file are incorrect: they are thus corrected with the lines below
if not(localComputer):
  alternativePathToFolderContainingOriginalVideos = "drive/MyDrive/rawVideos/"
  for video in videos:
    addAlternativePathToOriginalVideo(pathToZZoutput, video, alternativePathToFolderContainingOriginalVideos)

###

if __name__ == '__main__':

  results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])
  
  if generateInitialImages:
    for video in videos:
      createInitialImages(video, 'rolloverManualClassification.json', pathToZZoutput + '/', imagesToClassifyHalfDiameter, initialImagesFolder)

  if generateTrainingDataset:
    refreshTrainingDataset()
  cleanModel('model')

  # Creating learning dataset
  if generateTrainingDataset:
    trainingVid = videos.copy()
    testingVid  = trainingVid.pop(random.randint(0, len(videos)-1))
    print("Videos used for training:", trainingVid)
    print("Videos used for testing:", testingVid)
    for vidTrain in trainingVid:
      createTrainOrTestDataset(initialImagesFolder + '/', vidTrain, os.path.join('trainingDataset','train'), int(resizeCropDimension/2))
    createTrainOrTestDataset(initialImagesFolder + '/', testingVid, os.path.join('trainingDataset','val'), int(resizeCropDimension/2))
    print("Training and testing dataset created")
  
  # Transfert learning
  removeIpynbCheckpointsFromTrainingDataset()
  learnModel(epochsNbTraining, 'model', resizeCropDimension, learningParameters, showImagesUsedForTraining)
