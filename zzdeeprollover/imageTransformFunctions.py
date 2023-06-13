import cv2
import numpy as np

from torchvision import transforms
from PIL import Image
to_pil = transforms.ToPILImage()

def recenterImageOnEyes(img,window):
  
  blurwindow = 7 #31 # Must be an odd number
  rows = len(img)
  cols = len(img[0])
  grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(grey, (blurwindow, blurwindow),0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
  x = minLoc[0]
  y = minLoc[1]
    
  directTransfert = True
  
  ymin = y - window
  ymax = y + window
  xmin = x - window
  xmax = x + window
  
  yminDst = 0
  ymaxDst = 2 * window
  xminDst = 0
  xmaxDst = 2 * window
  
  if ymin < 0:
    directTransfert = False
    yminDst = -ymin
    ymin = 0
  if ymax > rows:
    directTransfert = False
    ymaxDst = 2 * window - (ymax - rows)
    ymax = rows
  if xmin < 0:
    directTransfert = False
    xminDst = -xmin
    xmin = 0
  if xmax > cols:
    directTransfert = False
    xmaxDst = 2 * window - (xmax - cols)
    xmax = cols
  
  if directTransfert:
    dst = img[ymin:ymax, xmin:xmax]
  else:
    dst = np.zeros((2 * window, 2 * window, 3), dtype='uint8')
    dst[:, :, :] = np.median(img)
    dst[yminDst:ymaxDst, xminDst:xmaxDst] = img[ymin:ymax, xmin:xmax]
    
  if ymin != y - window or ymax != y + window or xmin != x - window or xmax != x + window:
    print("WHAT:", ymin != y - window, ymax != y + window, xmin != x - window, xmax != x + window)
    # pil_image = to_pil(dst)
    # pil_image.show()
  
  return dst
