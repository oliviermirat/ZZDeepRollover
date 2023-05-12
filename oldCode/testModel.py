### Getting the data

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
from imageTransformFunctions import recenterImageOnEyes

def testModel(data_root, recenterImageWindow):

  classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
  feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
  image_data = image_generator.flow_from_directory(str(data_root))


  ### Getting the classifier

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

  # Testing model

  test_root = data_root

  directory_path = test_root + '/normal/'
  maxiter = 100000000
  iter = 0
  tot1 = []
  directory = os.fsencode(directory_path)
  for file in os.listdir(directory):
    name = os.fsdecode(file)
    if name.endswith(".png"): 
      filename = directory_path + name
      im = cv2.imread(filename)
      im1 = cv2.resize(im,(224,224))
      im1 = np.array(im1, dtype=np.float32) / 255.0
      if iter < maxiter:
        tot1.append(im1)
        iter = iter + 1

  directory_path = test_root + '/rollover/'
  iter = 0
  tot2 = []
  directory = os.fsencode(directory_path)
  for file in os.listdir(directory):
    name = os.fsdecode(file)
    if name.endswith(".png"): 
      filename = directory_path + name
      im = cv2.imread(filename)
      if recenterImageWindow:
        im = recenterImageOnEyes(im,recenterImageWindow)
      im1 = cv2.resize(im,(224,224))
      im1 = np.array(im1, dtype=np.float32) / 255.0
      if iter < maxiter:
        tot2.append(im1)
        iter = iter + 1
  tot = np.concatenate((tot1, tot2), axis=0)

  result = model.predict(tot)
  result.shape
  np.argmax(result, axis=-1)
  normalClassedAsRollo = np.sum(np.argmax(result[0:len(tot1)], axis=-1))
  rolloClassedAsRollo = np.sum(np.argmax(result[len(tot1):(len(tot1)+len(tot2))], axis=-1))
  print("normal classified as rollover:",normalClassedAsRollo,"out of",len(tot1),"; so:",(normalClassedAsRollo/len(tot1))*100,"%")
  print("rollover classified as rollover:",rolloClassedAsRollo,"out of",len(tot2),"; so:",(rolloClassedAsRollo/len(tot2))*100,"%")

if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

  data_root = sys.argv[1]
  recenterImageWindow = int(sys.argv[2])
  
  testModel(data_root, recenterImageWindow)
  