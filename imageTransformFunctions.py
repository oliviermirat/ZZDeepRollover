import cv2

def recenterImageOnEyes(img,window):
  blurwindow = 7 #31 # Must be an odd number
  rows = len(img)
  cols = len(img[0])
  grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(grey, (blurwindow, blurwindow),0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
  x = minLoc[0]
  y = minLoc[1]
  ymin = (y - window if y - window >= 0  else 0)
  ymax = (y + window if y + window <=rows else rows)
  xmin = (x - window if x - window >= 0  else 0)
  xmax = (x + window if x + window <=cols else cols)
  if ymin == 0:
    ymax = 2 * window
  if xmin == 0:
    xmax = 2 * window
  if ymax == rows:
    ymin = rows - 2 * window
  if xmax == cols:
    xmin = cols - 2 * window
  dst = img[ymin:ymax, xmin:xmax]
  return dst
