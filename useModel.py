from detectRolloverFrames import detectRolloverFrames
import random
import os

localComputer = True

file_path = 'listOfVideosToTakeIntoAccount.txt'
with open(file_path, 'r') as file:
  lines = file.readlines()
  videos = [line.strip() for line in lines]

testingVid  = videos.pop(random.randint(0, len(videos)-1))

pathToRawVideos = 'ZZoutputNew' if localComputer else 'drive/MyDrive/ZZoutputNew'

# Size of image on which DL algorithm will be applied
resizeCropDimension          = 34
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 50
# Window of median rolling mean applied on rollover detected
medianRollingMean = 5

pathToRawVideo  = os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi') if os.path.exists(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid + '.avi')) else os.path.join(os.path.join(os.path.join(pathToRawVideos, testingVid), testingVid), testingVid + '.seq')

if __name__ == '__main__':
  
  detectRolloverFrames(testingVid, pathToRawVideos + '/', medianRollingMean, resizeCropDimension, 1, 1, pathToRawVideo, imagesToClassifyHalfDiameter, os.path.join('model', 'model.pth'))
  