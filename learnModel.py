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

from numpy.random import randint

def newDataAugmentation(img, resizeSize):

  try:
    
    # Random rotation
    angle_rotation = randint(0,360)
    rows = len(img)
    cols = len(img[0])
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle_rotation,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    scaleD = int(cols/6)
    dst = dst[scaleD:(rows-scaleD), scaleD:(rows-scaleD)]

    # Random pixel intensity change
    changePixelIntensity = randint(-20, 20)
    dst = np.maximum(dst.astype(int)-changePixelIntensity,0).astype('uint8')
    
    # Random crop
    img = dst
    crop = randint(5, 10) #24
    rows = len(img)
    cols = len(img[0])
    xmin = crop
    ymin = crop
    xmax = cols - crop
    ymax = rows - crop
    img = img[ymin:ymax, xmin:xmax]
    dst = img
    
    # All images must be of the same size
    im1 = cv2.resize(dst, resizeSize)
    
    # cv2.imshow("im1", im1)
    # cv2.waitKey(0)
    
  except:
    im1 = 0
  
  return im1


def learnModel(data_root, epochsNb, modelFolder, newDataAugm):

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
  for image_batch,label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

  from tensorflow.python.keras import backend as K
  sess = K.get_session()
  init = tf.global_variables_initializer()

  sess.run(init)


  ### Transfer Learning

  def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)

  IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

  image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
  for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

  features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])

  features_extractor_layer.trainable = False

  model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
  ])
  model.summary()

  init = tf.global_variables_initializer()
  sess.run(init)

  ### Train the model

  model.compile(
    optimizer=tf.train.AdamOptimizer(), 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

  checkpoint_path = modelFolder + "/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create checkpoint callback
  cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)


  # if True:

  directory_path = data_root + '/normal/'
  tot1 = []
  directory = os.fsencode(directory_path)
  for file in os.listdir(directory):
    name = os.fsdecode(file)
    if name.endswith(".png"): 
      filename = directory_path + name
      im = cv2.imread(filename)
      if newDataAugm:
        im1 = newDataAugmentation(im, (224,224))
      else:
        im1 = cv2.resize(im,(224,224))
      if type(im1) != int:
        im1 = np.array(im1, dtype=np.float32) / 255.0
        tot1.append(im1)
  directory_path = data_root + '/rollover/'
  tot2 = []
  directory = os.fsencode(directory_path)
  for file in os.listdir(directory):
    name = os.fsdecode(file)
    if name.endswith(".png"): 
      filename = directory_path + name
      im = cv2.imread(filename)
      if newDataAugm:
        im1 = newDataAugmentation(im, (224,224))
      else:
        im1 = cv2.resize(im,(224,224))
      if type(im1) != int:
        im1 = np.array(im1, dtype=np.float32) / 255.0
        tot2.append(im1)
  tot = np.concatenate((tot1, tot2), axis=0)
  # tot = np.array(tot)

  n1 = len(tot1)
  n2 = len(tot2)
  lab1 = np.zeros((n1))
  lab2 = np.full((n2), 1)
  lab = np.concatenate((lab1, lab2), axis=0)
  from tensorflow.python.keras.utils import to_categorical
  y_binary = to_categorical(lab)

  model.fit(x=tot, y=y_binary, epochs=epochsNb, callbacks = [cp_callback])

  # else:
    
    # steps_per_epoch = image_data.samples//image_data.batch_size
    # model.fit((item for item in image_data), epochs=epochsNb, 
                      # steps_per_epoch=steps_per_epoch,
                      # callbacks = [cp_callback])

if __name__ == '__main__':

  __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

  data_root = sys.argv[1]
  epochsNb = int(sys.argv[2])
  modelFolder = sys.argv[3]
  
  if len(sys.argv) == 5:
    newDataAugm = int(sys.argv[4])
  else:
    newDataAugm = 0
  
  learnModel(data_root, epochsNb, modelFolder, newDataAugm)
