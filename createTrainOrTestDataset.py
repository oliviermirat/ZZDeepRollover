import cv2
import sys
import os, os.path
import random
import numpy as np
import sys
from imageTransformFunctions import recenterImageOnEyes

def createTrainOrTestDataset(inputFolder, videoName, outputFolder, nbRotations, recenter):

  def getNbElems(path):
    dirNormal   = path + '/normal'
    dirRollover = path + '/rollover'
    nbNormalImages = len([name for name in os.listdir(dirNormal) if os.path.isfile(os.path.join(dirNormal, name))])
    nbRolloverImages = len([name for name in os.listdir(dirRollover) if os.path.isfile(os.path.join(dirRollover, name))])
    minNbImages = min(nbNormalImages, nbRolloverImages)
    return [nbNormalImages, nbRolloverImages, minNbImages]
    
  def getAllImagesPath(src_path):
    imgPaths = []
    normal_images = os.listdir(src_path)
    for normal_image in normal_images:
      imgPath = os.path.join(src_path,normal_image)
      imgPaths.append(imgPath)
    return imgPaths

  def create_new_rotated_image(src_path, videoName, dst_path, nbRotations, k, recenter):
    kk = k
    img = cv2.imread(src_path)
    if recenter:
      img = recenterImageOnEyes(img,recenter)
    rows = len(img)
    cols = len(img[0])
    for i, angle_rotation in enumerate(np.arange(0,360,360/nbRotations)):
      M = cv2.getRotationMatrix2D((cols/2,rows/2),angle_rotation,1)
      dst = cv2.warpAffine(img,M,(cols,rows))
      scaleD = int(cols/6)
      dst = dst[scaleD:(rows-scaleD), scaleD:(rows-scaleD)]
      cv2.imwrite(dst_path+"/"+videoName+'_'+str(kk)+".png", dst)
      kk = kk + 1
    return kk

  def chooseImagesAndAugment(inputPath, videoName, outputPath, nbRotations, recenter):
    k = 0
    
    [nbNormalImages, nbRolloverImages, minNbImages] = getNbElems(inputPath + videoName)
    
    permutNormalImages   = np.random.permutation(nbNormalImages)
    permutRolloverImages = np.random.permutation(nbRolloverImages)
    
    imgNormalPaths = getAllImagesPath(inputPath + videoName + '/normal')
    imgRolloverPaths = getAllImagesPath(inputPath + videoName + '/rollover')
    
    dst_path_normal   = outputPath + '/normal'
    dst_path_rollover = outputPath + '/rollover'
    
    for i in range(0, minNbImages):
      image_path_normal   = imgNormalPaths[permutNormalImages[i]]
      image_path_rollover = imgRolloverPaths[permutRolloverImages[i]]
      k = create_new_rotated_image(image_path_normal, videoName, dst_path_normal, nbRotations, k, recenter)
      k = create_new_rotated_image(image_path_rollover, videoName, dst_path_rollover, nbRotations, k, recenter)

  chooseImagesAndAugment(inputFolder, videoName, outputFolder, nbRotations, recenter)
  
if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

  inputFolder = sys.argv[1]
  videoName = sys.argv[2]
  outputFolder = sys.argv[3]
  nbRotations = int(sys.argv[4])
  recenter = int(sys.argv[5])
  
  createTrainOrTestDataset(inputFolder, videoName, outputFolder, nbRotations, recenter)
  