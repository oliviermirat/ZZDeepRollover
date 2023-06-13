from zzdeeprollover.imageTransformFunctions import recenterImageOnEyes
import cv2
import sys
import os, os.path
import random
import numpy as np
import sys

def createTrainOrTestDataset(inputFolder, videoName, outputFolder, recenterImageWindow):
  
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

  def create_new_recentered_image(src_path, videoName, dst_path, k, recenter):
    kk = k
    img = cv2.imread(src_path)
    if recenter:
      img = recenterImageOnEyes(img,recenter)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dst_path+"/"+videoName+'_'+str(kk)+".png", img)
    kk = kk + 1
    return kk
  
  k = 0
  
  [nbNormalImages, nbRolloverImages, minNbImages] = getNbElems(inputFolder + videoName)
  
  permutNormalImages   = np.random.permutation(nbNormalImages)
  permutRolloverImages = np.random.permutation(nbRolloverImages)
  
  imgNormalPaths = getAllImagesPath(inputFolder + videoName + '/normal')
  imgRolloverPaths = getAllImagesPath(inputFolder + videoName + '/rollover')
  
  dst_path_normal   = outputFolder + '/normal'
  dst_path_rollover = outputFolder + '/rollover'

  for i in range(0, minNbImages):
    image_path_normal   = imgNormalPaths[permutNormalImages[i]]
    image_path_rollover = imgRolloverPaths[permutRolloverImages[i]]
    k = create_new_recentered_image(image_path_normal, videoName, dst_path_normal, k, recenterImageWindow)
    k = create_new_recentered_image(image_path_rollover, videoName, dst_path_rollover, k, recenterImageWindow)
