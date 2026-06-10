import cv2 as cv

def rescaleFrame(frame, scale = 0.75):
  # Images, Videos and Live Videos
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)

  dimensions = (width, height)

  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(capture, width, height):
  # Live videos only
  capture.set(3, width)
  capture.set(4, height)

def readImage():
  img = cv.imread('../media/insect.jpg')
  resizedImage = rescaleFrame(img)
  
  cv.imshow('ImageWindow', img)
  cv.imshow('rescaledImage', resizedImage)
  cv.waitKey(0)

def readVideo():
  capture = cv.VideoCapture('../media/clip.mp4')
  while True:
    isTrue, frame = capture.read()
    resizedFrame = rescaleFrame(frame)

    if not isTrue:
      break
    cv.imshow('videoWindow', frame)
    cv.imshow('rescaled_frame', resizedFrame)

    # Pressing 'D' closes the windows
    if cv.waitKey(20) & 0xFF == ord('d'):
      break
  capture.release()
  cv.destroyAllWindows()

readVideo()
readImage()
