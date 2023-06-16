from torchvision import transforms
from PIL import Image
import numpy as np
import random
import cv2

def rotate_image(image):
  if True:
    width, height = image.size
    image = np.array(image)
    resizeCrop = len(image)
    bigImg = np.zeros((width + 2*resizeCrop, height + 2*resizeCrop, 3), dtype='uint8')
    bigImg[:, :, :] = np.median(image)
    bigImg[resizeCrop:len(bigImg[0])-resizeCrop, resizeCrop:len(bigImg)-resizeCrop] = image
    angle = random.uniform(0, 360)
    rows = len(bigImg)
    cols = len(bigImg[0])
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    bigImg = cv2.warpAffine(bigImg, M, (cols,rows))
    return Image.fromarray(bigImg[resizeCrop:len(bigImg)-resizeCrop, resizeCrop:len(bigImg[0])-resizeCrop])
  else:
    angle = random.uniform(0, 360)
    return image.rotate(angle)
  
def crop_image(image, maxCrop):
  if maxCrop:
    resizeCrop = random.randint(0, maxCrop)
    width, height = image.size
    return image.crop((resizeCrop, resizeCrop, width-resizeCrop, height-resizeCrop))
  else:
    return image

class RotateTransformCustom:
    def __init__(self):
        pass

    def __call__(self, image):
        return rotate_image(image)

class CropTransformCustom:
    def __init__(self, maxCrop):
        self.maxCrop = maxCrop

    def __call__(self, image):
        return crop_image(image, self.maxCrop)

def get_data_transforms(resizeCropDimension, learningParameters={}):

  maxCrop               = learningParameters['maxCrop']               if 'maxCrop' in learningParameters else 0
  brightness_limit      = learningParameters['brightness_limit']      if 'brightness_limit' in learningParameters else 0
  contrast_limit        = learningParameters['contrast_limit']        if 'contrast_limit' in learningParameters else 0
  invert_probability    = learningParameters['invert_probability']    if 'invert_probability' in learningParameters else 0
  sharpness_probability = learningParameters['sharpness_probability'] if 'sharpness_probability' in learningParameters else 0
  
  data_transforms = {
    'train': transforms.Compose([
      RotateTransformCustom(),
      CropTransformCustom(maxCrop),
      transforms.ColorJitter(brightness_limit, contrast_limit),
      transforms.RandomInvert(invert_probability),
      transforms.RandomAdjustSharpness(sharpness_probability),
      transforms.Resize(resizeCropDimension),
      transforms.ToTensor(),
      transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
      transforms.Resize(resizeCropDimension),
      transforms.ToTensor(),
      transforms.Normalize([0.485], [0.229])
    ]),
  }
  
  return data_transforms
