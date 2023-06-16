from torchvision import transforms
import random

def rotate_image(image):
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
