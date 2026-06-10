import cv2 as cv
import numpy as np
import os
 
DIR = r'../data'
people = []

for person in os.listdir(DIR):
  people.append(person)

haar_classifier = cv.CascadeClassifier('../haar_cascade.xml')

features = [] 
labels = []

def create_train():
  for person in people:
    path = os.path.join(DIR, person)
    label = people.index(person)

    for img in os.listdir(path):
      img_path = os.path.join(path, img)

      img_array = cv.imread(img_path)
      if img_array is None:
        continue
      gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

      faces_rect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

      for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        features.append(faces_roi)
        labels.append(label)

create_train()

# Image training completed

features = np.array(features, dtype='object')
labels = np.array(labels)

# Training the recognizer on images and lables

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)

print("Training done!")