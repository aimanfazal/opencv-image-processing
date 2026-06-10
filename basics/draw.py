import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('blankWindow', blank)

class myDrawings:
  def paintImage():
    blank[200:300, 300:400] = 255, 0, 255
    cv.imshow('coloured', blank)

  def drawRectangle():
    #* try thickness = -1 or cv.FILLED
    # cv.rectangle(blank, (0,0), (500,100), (0,0,255), thickness=4)
    cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,0,255), thickness=4)
    cv.imshow('rectangleWindow', blank)

  def drawCircle():
    cv.circle(blank, (250,250), 200, (0, 0, 255), thickness=4)
    cv.imshow('circleWindow', blank)

  def drawLine(start_point = (0,0), end_point = (250,250)):
    cv.line(blank, start_point, end_point, (0, 150,80), thickness=5)
    cv.imshow('lineWindow', blank)

  def writeText():
    cv.putText(blank, 'OpenCV', (250, 250), cv.FONT_HERSHEY_TRIPLEX, 1.0, (12, 300, 12), 1)
    cv.imshow('textWindow', blank)

d = myDrawings
d.drawRectangle()

cv.waitKey(0)
