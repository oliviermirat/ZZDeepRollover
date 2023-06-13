import json
import os

def addAlternativePathToOriginalVideo(pathToZZoutput, video, alternativePathToFolderContainingOriginalVideos):

  with open(os.path.join(os.path.join(pathToZZoutput, video), 'results_' + video + '.txt'), 'r') as file:
    jsonFile = json.load(file)
  
  if os.path.exists(os.path.join(os.path.join(alternativePathToFolderContainingOriginalVideos, video), video + '.avi')):
    alternativePathToOriginalVideo = os.path.join(os.path.join(alternativePathToFolderContainingOriginalVideos, testingVid), testingVid + '.avi') 
  else:
    alternativePathToOriginalVideo = os.path.join(os.path.join(alternativePathToFolderContainingOriginalVideos, video), video + '.seq')
  
  jsonFile['alternativePathToOriginalVideo'] = alternativePathToOriginalVideo
  
  with open(os.path.join(os.path.join(pathToZZoutput, video), 'results_' + video + '.txt'), 'w') as file:
    json.dump(jsonFile, file)
