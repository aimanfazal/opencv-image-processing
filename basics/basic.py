import cv2 as cv

def rescaleFrame(frame, scale = 0.75):
  # Images, Videos and Live Videos
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)

  dimensions = (width, height)

  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_sample = cv.imread('../media/cat.jpg')

#Resized Image
img = rescaleFrame(img_sample, 0.1)

# To Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)

# Edge Cascade
canny = cv.Canny(img, 125, 190)

# Dilating
dilated = cv.dilate(img, (15,15), iterations=15)

# Eroding
eroded = cv.erode(dilated, (5, 5), iterations=5)

# Resize
resize = cv.resize(img, (155, 155), interpolation=cv.INTER_CUBIC)

# Cropping
cropped = img[50:200, 50:450]

cv.imshow('sample', cropped)
cv.waitKey(0)