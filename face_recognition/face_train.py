import cv2 as cv
import numpy as np
import os

DIR = r'../data'
FACE_SIZE = (250, 250)

people = sorted(os.listdir(DIR))

haar_classifier = cv.CascadeClassifier('../haar_cascade.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        print(f'Training {person} ({label})')

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img_array = cv.imread(img_path)

            if img_array is None:
                print(f'Could not read: {img_path}')
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

            if len(faces_rect) == 0:
                print(f'No face found: {img_path}')
                continue

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]

                # Normalize size
                face_roi = cv.resize(face_roi, FACE_SIZE)

                features.append(face_roi)
                labels.append(label)

create_train()

print(f'\nTraining samples collected: {len(features)}')

labels = np.array(labels, dtype=np.int32)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', np.array(features))
np.save('labels.npy', labels)

print("Training complete!")