import cv2 as cv
import os

def rescaleFrame(frame, scale = 0.5):
  # Images, Videos and Live Videos
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)

  dimensions = (width, height)

  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

haar_classifier = cv.CascadeClassifier('../haar_cascade.xml')

people = []
for person in os.listdir(r'../data'):
  people.append(person)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Test Image
# img_path = r'../test_files/fr_2.jpg'
img_path = r'C:\Users\icxnicAF\Downloads\test.jpg'
img = cv.imread(img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the face in the image
faces_rect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

for (x, y, w, h) in faces_rect:
  faces_roi = gray[y:y+h, x:x+w]
  label, confidence = face_recognizer.predict(faces_roi)

  print(f"Label = {people[label]} with a confidence of {confidence}\n")

  print("People:", people)
  print("Detected faces:", len(faces_rect))
  print("ROI shape:", faces_roi.shape)


  cv.putText(img, str(people[label]), (x//2, y//2), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=3)
  cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=3)

gray = rescaleFrame(gray)
img = rescaleFrame(img)

cv.imshow('Sample_Image', gray)
cv.imshow('Detected_Face', img)

cv.waitKey(0)

# Confidence Value Range:

# 0  – 20	Very strong match
# 20 – 50	Good match
# 50 – 80	Weak match
# 80+ -	  Likely wrong person