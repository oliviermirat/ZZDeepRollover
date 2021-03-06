from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import layers
import numpy as np
import os
tf.VERSION
import cv2
import sys
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from imageTransformFunctions import recenterImageOnEyes
from createValidationVideo import createValidationVideo
# import pdb

def detectRolloverFrames(data_root, videoName, path, medianRollingMean, recenterImageWindow, comparePredictedResultsToManual, validationVideo):

  if (medianRollingMean % 2 == 0):
    sys.exit("medianRollingMean must be an odd number")

  ### Loading the classifier

  classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
  feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
  image_data = image_generator.flow_from_directory(str(data_root))

  def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)
    
  IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))

  classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
  classifier_model = tf.keras.Sequential([classifier_layer])
  classifier_model.summary()

  image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)

  from tensorflow.python.keras import backend as K
  sess = K.get_session()
  init = tf.global_variables_initializer()

  sess.run(init)

  ####

  def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)

  IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

  features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])

  features_extractor_layer.trainable = False

  model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
  ])
  model.summary()

  init = tf.global_variables_initializer()
  sess.run(init)

  ### 

  model.compile(
    optimizer=tf.train.AdamOptimizer(), 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

  checkpoint_path = "model/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  model.load_weights(checkpoint_path)


  ### Loading the images and applying the classifier on them

  videoPath = path + videoName + '/results_' + videoName + '.txt'

  if (os.path.isfile(videoPath)):
    
    # Applying rollover classifier to each frame and saving the results in a txt file
    
    file = open(videoPath, 'r')
    j = json.loads(file.read())
    wellPoissMouv = j['wellPoissMouv']
    wellPositions = j['wellPositions']
    nbWell = len(wellPositions)
    rolloversAllWells = []
    rolloversMedFiltAllWells = []
    rolloverPercentageAllWells = []
    # going through each well in super structure
    for i in range(0,nbWell):
      xwell = wellPositions[i]['topLeftX']
      ywell = wellPositions[i]['topLeftY']
      if xwell < 0:
        xwell = 0
      if ywell < 0:
        ywell = 0
      videoPath2 = path+videoName+'/'+videoName+'_'+str(i+1)+'/'+videoName+'_'+str(i+1)+'.avi'
      if (len(wellPoissMouv[i])):
        if (len(wellPoissMouv[i][0])):
          cap = cv2.VideoCapture(videoPath2)
          videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          rollovers = np.zeros((videoLength))
          rolloverPercentage = np.zeros((videoLength))
          frames = []
          framesNumber = []
          nbMouv = len(wellPoissMouv[i][0])
          # going through each movement for the well
          for j in range(0,nbMouv):
            if (len(wellPoissMouv[i][0][j])):
              item = wellPoissMouv[i][0][j]
              BoutStart = item['BoutStart']
              BoutEnd   = item['BoutEnd']
              k = BoutStart
              cap.set(cv2.CAP_PROP_POS_FRAMES,BoutStart)
              while (k <= BoutEnd):
                ret, frame = cap.read()
                if ret == True:
                  if recenterImageWindow:
                    frame = recenterImageOnEyes(frame,recenterImageWindow)
                  rows = len(frame)
                  cols = len(frame[0])
                  scaleD = int(cols/6)
                  frame = frame[scaleD:(rows-scaleD), scaleD:(rows-scaleD)]
                  frame = cv2.resize(frame,(224,224))
                  frame = np.array(frame, dtype=np.float32) / 255.0
                  frames.append(frame)
                  framesNumber.append(k)
                else:
                  break
                k = k + 1
      frames = np.array(frames)
      resultRaw = model.predict(frames)
      result = np.argmax(resultRaw, axis=-1)
      rollovers[framesNumber] = result
      rolloverPercentage[framesNumber] = resultRaw[:,1]
      rolloversMedFiltSeries = (pd.Series(rollovers)).rolling(medianRollingMean).median()
      for i in range(0, medianRollingMean):
        rolloversMedFiltSeries[i] = 0
      rolloversMedFiltSeries = np.roll(rolloversMedFiltSeries,int(-((medianRollingMean-1)/2)))
      rolloversAllWells.append(rollovers)
      rolloversMedFiltAllWells.append(rolloversMedFiltSeries)
      rolloverPercentageAllWells.append(rolloverPercentage)
    rolloversAllWells = np.array(rolloversAllWells)
    rolloversMedFiltAllWells = np.array(rolloversMedFiltAllWells)
    cap.release()
    np.savetxt(path + videoName + '/rolloverClassified.txt', rolloversMedFiltAllWells, fmt='%d')
    np.savetxt(path + videoName + '/rolloverPercentages.txt', rolloverPercentageAllWells, fmt='%f')
    
    if validationVideo:
      createValidationVideo(videoName, path, rolloversMedFiltAllWells, rolloverPercentageAllWells)
    
    if comparePredictedResultsToManual:
    
      # Printing rollovers frames before the rolling median filter
      
      rollForAllWells = []
      for rollover in rolloversAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWells.append(rollForCurWell)
      
      print("Before rolling medfilter:")
      print(rollForAllWells)
      
      # Printing the rollover after the rolling median filter

      rollForAllWellsMed = []
      for rollover in rolloversMedFiltAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWellsMed.append(rollForCurWell)
      
      print("\nAfter rolling medfilter:")
      print(rollForAllWellsMed)
      
      # Printing manual classifications of rollovers
      
      print("\nManual classification:")
      rolloverFrameFile = path + videoName + '/rolloverManualClassification.json'
      fileRollover = open(rolloverFrameFile, 'r')
      rolloverFrame = json.loads(fileRollover.read())
      
      trueRolloverAllWells = []
      for well in rolloverFrame:
        trueRollovers = np.zeros((videoLength))
        for boundaries in rolloverFrame[well]['rollover']:
          left  = boundaries[0]
          right = boundaries[1]
          for i in range(left,right+1):
            trueRollovers[i] = 1
        trueRolloverAllWells.append(trueRollovers)
      trueRolloverAllWells = np.array(trueRolloverAllWells)
      
      wellNumberAllWells = []
      for idx, well in enumerate(rolloverFrame):
        wellNumber = np.zeros((videoLength))
        # for boundaries in rolloverFrame[well]['rollover']:
          # left  = boundaries[0]
          # right = boundaries[1]
          # for i in range(left,right+1):
            # wellNumber[i] = idx+1
        for idx2, aaa in enumerate(wellNumber):
          wellNumber[idx2] = idx+1
        wellNumberAllWells.append(wellNumber)
      wellNumberAllWells = np.array(wellNumberAllWells)
      
      boutNumberAllWells = []
      for idx, well in enumerate(rolloverFrame):
        boutNumber = np.zeros((videoLength))
        # for boundaries in rolloverFrame[well]['rollover']:
          # left  = boundaries[0]
          # right = boundaries[1]
          # for i in range(left,right+1):
            # boutNumber[i] = i
        for idx2, aaa in enumerate(boutNumber):
          boutNumber[idx2] = idx2
        boutNumberAllWells.append(boutNumber)
      boutNumberAllWells = np.array(boutNumberAllWells)
      
      rollForAllWellsTrue = []
      for rollover in trueRolloverAllWells:
        rollForCurWell = []
        for idx, roll in enumerate(rollover):
          if roll:
            rollForCurWell.append(idx)
        rollForAllWellsTrue.append(rollForCurWell)
      
      print(rollForAllWellsTrue)
      print("")
      print(rolloverFrame)
      
      # Getting the frames in-between "rollovers" and "normal"
      
      inBetweensAllWells = []
      for well in rolloverFrame:
        inBetweens = np.zeros((videoLength))
        for boundaries in rolloverFrame[well]['inBetween']:
          left  = boundaries[0]
          right = boundaries[1]
          for i in range(left,right+1):
            inBetweens[i] = i
        inBetweens = inBetweens.tolist()
        inBetweensAllWells.append(inBetweens)
        
      # Removing the "in-between" frames from the before medfilt rollovers, after medfilt rollovers, and the trues rollovers
      
      rolloversMedFiltAllWellsInBetweenRemoved = []
      rolloversAllWellsInBetweenRemoved        = []
      trueRolloverAllWellsInBetweenRemoved     = []
      wellNumberAllWellsInBetweenRemoved       = []
      boutNumberAllWellsInBetweenRemoved       = []
      # pdb.set_trace()
      for idx, inBetweens in enumerate(inBetweensAllWells):
        rolloversMedFiltAllWellsInBetweenRemoved.extend(np.delete(rolloversMedFiltAllWells[idx], inBetweens).tolist())
        rolloversAllWellsInBetweenRemoved.extend(np.delete(rolloversAllWells[idx], inBetweens).tolist())
        trueRolloverAllWellsInBetweenRemoved.extend(np.delete(trueRolloverAllWells[idx], inBetweens).tolist())
        wellNumberAllWellsInBetweenRemoved.extend(np.delete(wellNumberAllWells[idx], inBetweens).tolist())
        boutNumberAllWellsInBetweenRemoved.extend(np.delete(boutNumberAllWells[idx], inBetweens).tolist())
      rolloversMedFiltAllWellsInBetweenRemoved = np.array(rolloversMedFiltAllWellsInBetweenRemoved)
      rolloversAllWellsInBetweenRemoved        = np.array(rolloversAllWellsInBetweenRemoved)
      trueRolloverAllWellsInBetweenRemoved     = np.array(trueRolloverAllWellsInBetweenRemoved)
      wellNumberAllWellsInBetweenRemoved       = np.array(wellNumberAllWellsInBetweenRemoved)
      boutNumberAllWellsInBetweenRemoved       = np.array(boutNumberAllWellsInBetweenRemoved)
      
      # Using scikit learn to print classification statistics
      
      print(classification_report(trueRolloverAllWellsInBetweenRemoved, rolloversAllWellsInBetweenRemoved))
      print(classification_report(trueRolloverAllWellsInBetweenRemoved, rolloversMedFiltAllWellsInBetweenRemoved))
      
      print(precision_recall_fscore_support(trueRolloverAllWellsInBetweenRemoved, rolloversAllWellsInBetweenRemoved))
      print(precision_recall_fscore_support(trueRolloverAllWellsInBetweenRemoved, rolloversMedFiltAllWellsInBetweenRemoved))
      
      # Printing "normal classified as rollover" percentage and "rollover classified as rollover" percentage and returning those two values
      
      normalClassedAsRollo = 0
      rolloClassedAsRollo = 0
      totalTrueNormal = 0
      totalTrueRollo  = 0
      for idx,true in enumerate(trueRolloverAllWellsInBetweenRemoved):
        pred = rolloversMedFiltAllWellsInBetweenRemoved[idx]
        if true == 0:
          totalTrueNormal = totalTrueNormal + 1
          if pred == 1:
            normalClassedAsRollo = normalClassedAsRollo + 1
            print("False positive: well:",wellNumberAllWellsInBetweenRemoved[idx]," ; bout:",boutNumberAllWellsInBetweenRemoved[idx])
        if true == 1:
          totalTrueRollo = totalTrueRollo + 1
          if pred == 1:
            rolloClassedAsRollo = rolloClassedAsRollo + 1
          else:
            print("False negative: well:",wellNumberAllWellsInBetweenRemoved[idx]," ; bout:",boutNumberAllWellsInBetweenRemoved[idx])
      
      if totalTrueNormal:
        print("normal classified as rollover:",normalClassedAsRollo,"out of",totalTrueNormal,"; so:",(normalClassedAsRollo/totalTrueNormal)*100,"%")
      else:
        print("no true normal in this dataset")
      if totalTrueRollo:
        print("rollover classified as rollover:",rolloClassedAsRollo,"out of",totalTrueRollo,"; so:",(rolloClassedAsRollo/totalTrueRollo)*100,"%")
      else:
        print("no true rollovers in this dataset")
        
      return [normalClassedAsRollo, totalTrueNormal, rolloClassedAsRollo, totalTrueRollo]


if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

  data_root = sys.argv[1]
  videoName = sys.argv[2]
  path = sys.argv[3]
  medianRollingMean = int(sys.argv[4])
  recenterImageWindow = int(sys.argv[5])
  comparePredictedResultsToManual = int(sys.argv[6])
  validationVideo = int(sys.argv[7])

  detectRolloverFrames(data_root, videoName, path, medianRollingMean, recenterImageWindow, comparePredictedResultsToManual, validationVideo)
  