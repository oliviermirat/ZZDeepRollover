from torchvision import transforms
import random

def rotate_image(image):
  angle = random.uniform(0, 360)
  return image.rotate(angle)

def get_data_transforms(resizeCropDimension):
  
  data_transforms = {
    'train': transforms.Compose([
      transforms.Lambda(rotate_image),
      # transforms.HorizontalFlip(),
      # transforms.VerticalFlip(),
      # transforms.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
      transforms.Resize(resizeCropDimension),
      # transforms.RandomResizedCrop(randomResizeCropDimension),
      transforms.ToTensor(),
      transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
      transforms.Resize(resizeCropDimension),
      # transforms.CenterCrop(randomResizeCropDimension),
      transforms.ToTensor(),
      transforms.Normalize([0.485], [0.229])
    ]),
  }
  
  return data_transforms
