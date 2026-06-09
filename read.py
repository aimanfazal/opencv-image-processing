import cv2 as cv

def readImage():
  img = cv.imread('imgs/insect.jpg')
  cv.imshow('ImageWindow', img)
  cv.waitKey(0)

def readVideo():
  capture = cv.VideoCapture('vids/clip.mp4')
  while True:
    isTrue, frame = capture.read()

    if not isTrue:
      break
    cv.imshow('videoWindow', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
      break
  capture.release()
  cv.destroyAllWindows()

readImage()
readVideo()