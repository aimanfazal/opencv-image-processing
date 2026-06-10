import cv2 as cv

def readImage():
  img = cv.imread('../media/insect.jpg')
  cv.imshow('Image_Window', img)
  cv.waitKey(0)

def readVideo():
  capture = cv.VideoCapture('../media/clip.mp4')
  while True:
    isTrue, frame = capture.read()

    if not isTrue:
      break
    cv.imshow('videoWindow', frame)

    # Pressing 'D' closes the media windows
    if cv.waitKey(20) & 0xFF == ord('d'):
      break
  capture.release()
  cv.destroyAllWindows()

readImage()
readVideo()