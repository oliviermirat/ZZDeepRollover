from createInitialImages import createInitialImages
from createTrainOrTestDataset import createTrainOrTestDataset
from learnModel import learnModel
from testModel import testModel
from detectRolloverFrames import detectRolloverFrames
import subprocess
from subprocess import Popen
import numpy as np
import pandas as pd

windows = 0 # 1 on windows platform, 0 on linux

pathToRawVideos = 'ZZoutput/'

recenterImageWindow=12
numberOfRotationsDataAugmentationTraining=12
numberOfRotationsDataAugmentationTesting=1
epochsNbTraining=10
medianRollingMean=5

videos = ['20181006-1-2_1', '20181006-1-2_4', '20181006-2-2_5', '4-2_4', '20181006-1-2_7', '20181006-2-2_1', '20181006-3-2_1', '20181006-3-2_6']
# these videos have no rollovers: 20181006-2-2_2 and 20181006-3-2_2

results = pd.DataFrame(np.zeros((len(videos), 6)), index=videos, columns=['normalClassedAsRolloNb', 'totalNbTrueNormal', 'rolloClassedAsRolloNb', 'totalNbTrueRollo', 'normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent'])

for video in videos:
  createInitialImages(video,'rolloverManualClassification.json',pathToRawVideos)

trainingVid = videos.copy()
testingVid = trainingVid.pop(0)

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

normalClassedAsRolloPercent  = (normalClassedAsRollo/totalTrueNormal) * 100
rolloClassedAsRolloNbPercent = (rolloClassedAsRollo/totalTrueRollo) * 100

results.loc[testingVid] = [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo, normalClassedAsRolloPercent, rolloClassedAsRolloNbPercent]

print("Final Results")
df = results[['normalClassedAsRolloPercent', 'rolloClassedAsRolloNbPercent']]
print(df)
print("")
print("normalClassedAsRolloPercent average:", df["normalClassedAsRolloPercent"].mean(), "%")
print("rolloClassedAsRolloNbPercent average:", df["rolloClassedAsRolloNbPercent"].mean(), "%")
