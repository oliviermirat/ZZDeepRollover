from zzdeeprollover.addAlternativePathToOriginalVideo import addAlternativePathToOriginalVideo
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

if True:
  learningParameters = {}
else:
  learningParameters = {'maxCrop': 1, 'brightness_limit': 0.05, 'contrast_limit': 0.05, 'invert_probability': 0, 'sharpness_probability': 0}

showImagesUsedForTraining = False

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

  for idx, video in enumerate(videos):

    trainingVid = videos.copy()
    testingVid  = trainingVid.pop(idx)
    
    print("Videos used for training:", trainingVid)
    print("Videos used for testing:", testingVid)
    
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
    learnModel(epochsNbTraining, 'model', resizeCropDimension, learningParameters, showImagesUsedForTraining)
    
    # Testing the model on the entire video
    validationVideo = 1
    [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo] = detectRolloverFrames(testingVid, pathToZZoutput + '/', medianRollingMean, resizeCropDimension, 1, validationVideo, imagesToClassifyHalfDiameter, os.path.join('model', 'model.pth'))
    
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
