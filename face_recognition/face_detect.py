import cv2 as cv

def rescaleFrame(frame, scale = 0.75):
  # Images, Videos and Live Videos
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)

  dimensions = (width, height)

  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

source = cv.imread('../test_files/1.jpg')
# source = rescaleFrame(source)
gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

haar_classifier = cv.CascadeClassifier('../haar_cascade.xml')

#* greater number of minNeighbours, higher accuracy
face_detect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f"Number of face(s) detected: {len(face_detect)}")

for (x,y,w, h) in face_detect:
  cv.rectangle(source, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('detected_faces', source)

cv.waitKey(0)