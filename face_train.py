import cv2 as cv
import numpy as np
import os
 
people = []
DIR = r'path'

haar_classifier = cv.CascadeClassifier('haar_cascade.xml')

features = []
labels = []

def create_train():
  for person in people:
    path = os.path.join(DIR, person)
    label = people.index(person)

    for img in os.listdir(path):
      img_path = os.path.join(path, img)

      img_array = cv.imread(img_path)
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

      faces_rect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

      for x,y,w,h in faces_rect:
        faces_roi = gray[y:y+w, x:x+w]
        features.append(faces_roi)
        labels.append(label)