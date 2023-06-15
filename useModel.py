from zzdeeprollover.addAlternativePathToOriginalVideo import addAlternativePathToOriginalVideo
from zzdeeprollover.detectRolloverFrames import detectRolloverFrames
import random
import os

localComputer = True

# Size of image on which DL algorithm will be applied
resizeCropDimension          = 34
# Approximate half dimension of validation video and of initial image extracted
imagesToClassifyHalfDiameter = 50
# Window of median rolling mean applied on rollover detected
medianRollingMean = 5

pathToZZoutput = 'ZZoutputNew' if localComputer else 'drive/MyDrive/ZZoutputNew'

file_path = 'listOfVideosToTakeIntoAccount.txt'
with open(file_path, 'r') as file:
  lines = file.readlines()
  videos = [line.strip() for line in lines]

testingVid  = videos.pop(random.randint(0, len(videos)-1))

# If launched on a computer other than the one used to launch the tracking, the paths to the original raw videos saved in the result file are incorrect: they are thus corrected with the lines below
if not(localComputer):
  alternativePathToFolderContainingOriginalVideos = "drive/MyDrive/rawVideos/"
  for video in videos:
    addAlternativePathToOriginalVideo(pathToZZoutput, video, alternativePathToFolderContainingOriginalVideos)

###

if __name__ == '__main__':

  print("Videos used for testing:", testingVid)
  
  detectRolloverFrames(testingVid, pathToZZoutput + '/', medianRollingMean, resizeCropDimension, 1, 1, imagesToClassifyHalfDiameter, os.path.join('model', 'model.pth'))
