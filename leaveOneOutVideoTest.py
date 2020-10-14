from createInitialImages import createInitialImages
from createTrainOrTestDataset import createTrainOrTestDataset
from learnModel import learnModel
from testModel import testModel
from detectRolloverFrames import detectRolloverFrames
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd

windows = 1 # set to 1 on windows platform, 0 on linux

pathToRawVideos = 'ZZoutput/'

recenterImageWindow=24
numberOfRotationsDataAugmentationTraining=12
numberOfRotationsDataAugmentationTesting=1
epochsNbTraining=15
medianRollingMean=5

videos = ['20190727-13-2-1', '20190427-1-2-5', '20190503-2-2-3', '20190727-6-2-4', '20190503-1-2-5', '20190427-3-2-2', '20190427-1-2-2', '20190427-4-2-1', '20190503-4-2-7', '20190727-2-2-1', '20190727-9-2-7', '20190427-1-2-1', '20190427-5-2-1', '20190727-10-2-7', '20190727-3-2-2', '20190427-2-2-6', '20190727-11-2-7', '20190727-4-2-6', '20190427-2-2-8', '20190503-2-2-1', '20190727-1-2-3', '20190727-5-2-7']
# these videos have no rollovers: 20181006-2-2_2 and 20181006-3-2_2

results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])

for video in videos:
  createInitialImages(video,'rolloverManualClassification.json',pathToRawVideos)

for idx, video in enumerate(videos):

  trainingVid = videos.copy()
  testingVid = trainingVid.pop(idx)
  
  print("\nTestingVid:", testingVid, "\n")

  if windows:
    print("windows")
    Process=Popen('cleanDataset.sh %s' % (str('trainingDataset')), shell=True)
  else:
    print("linux")
    Process=Popen('./cleanDataset.sh %s' % (str('trainingDataset')), shell=True)
  Process.wait()

  # Creating learning dataset. Data augmenation done: 10 rotations.
  for vidTrain in trainingVid:
    createTrainOrTestDataset('initialImages/',vidTrain,'trainingDataset',numberOfRotationsDataAugmentationTraining,recenterImageWindow)

  if windows:
    Process=Popen('cleanDataset.sh %s' % (str('testDataset')), shell=True)
  else:
    Process=Popen('./cleanDataset.sh %s' % (str('testDataset')), shell=True)
  Process.wait()
  # Creating testing dataset. 1 rotations: no data augmentation done.
  # createTrainOrTestDataset('initialImages/',testingVid,'testDataset',numberOfRotationsDataAugmentationTesting,recenterImageWindow)

  if windows:
    Process=Popen('cleanModel.sh %s' % (str('model')), shell=True)
  else:
    Process=Popen('./cleanModel.sh %s' % (str('model')), shell=True)
  # Transfert learning
  learnModel('trainingDataset', epochsNbTraining, 'model')

  # Testing the model on the testing dataset
  # testModel('testDataset', recenterImageWindow)

  # Testing the model on the entire video
  validationVideo = 1
  [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo] = detectRolloverFrames('trainingDataset',testingVid,pathToRawVideos,medianRollingMean,recenterImageWindow, 1, validationVideo)
  
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
