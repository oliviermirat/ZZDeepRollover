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

# Size of image on which DL algorithm will be applied
resizeCropDimension          = 68
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 100
# Number of epoch for DL training
epochsNbTraining=10 #15
# Window of median rolling mean applied on rollover detected
medianRollingMean=5

videos = ['4wellsZebrafishLarvaeEscapeResponses_1', '4wellsZebrafishLarvaeEscapeResponses_2']
#videos = ['20190727-13-2-1', '20190427-1-2-5', '20190503-2-2-3', '20190727-6-2-4', '20190503-1-2-5', '20190427-3-2-2', '20190427-1-2-2', '20190427-4-2-1', '20190503-4-2-7', '20190727-2-2-1', '20190727-9-2-7', '20190427-1-2-1', '20190427-5-2-1', '20190727-10-2-7', '20190727-3-2-2', '20190427-2-2-6', '20190727-11-2-7', '20190727-4-2-6', '20190427-2-2-8', '20190503-2-2-1', '20190727-1-2-3', '20190727-5-2-7']

pathToRawVideos = 'ZZoutput'

###

if __name__ == '__main__':

  results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])
  
  if generateInitialImages:
    for video in videos:
      createInitialImagesNewZZ(video, 'rolloverManualClassification.json', pathToRawVideos + '/', imagesToClassifyHalfDiameter)


  trainingVid = videos.copy()

  if generateTrainingDataset:
    refreshTrainingDataset()
  cleanModel('model')

  # Creating learning dataset
  for vidTrain in trainingVid:
    createTrainOrTestDataset('initialImages/', vidTrain, os.path.join('trainingDataset','train'), int(resizeCropDimension/2))

  # Transfert learning
  removeIpynbCheckpointsFromTrainingDataset()
  learnModel(epochsNbTraining, 'model', resizeCropDimension)
