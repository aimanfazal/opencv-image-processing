import cv2 as cv
import numpy as np
  
def readImage(path):
  sampleImage = cv.imread(path)
  cv.imshow('theWindow', sampleImage)
  cv.waitKey(0)

def transform(image, x, y):
  transMat = np.float32([[1, 0, x], [0, 1, y]])
  dimensions = (image.shape[1], image.shape[0])
  return cv.warpAffine(image, transMat, dimensions)

def rotation(image, angle, rotPoint=None):
  (heigt, width) = image.shape[:2]

  if rotPoint is None:
    rotPoint = (width//2, heigt//2)
  
  rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
  dimensions = (width, heigt)

  return cv.warpAffine(image, rotMat, dimensions)

readImage('imgs/farm.jpg')
