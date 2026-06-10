import cv2 as cv
import os

FACE_SIZE = (250, 250)
UNKNOWN_THRESHOLD = 55

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

haar_classifier = cv.CascadeClassifier('../haar_cascade.xml')

people = sorted(os.listdir(r'../data'))

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Test image
img_path = r'..\test_files\recognition_images\fr_14.jpg'

img = cv.imread(img_path)

if img is None:
    raise FileNotFoundError(img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

print(f"Detected faces: {len(faces_rect)}")

if len(faces_rect) == 0:
    print("No face detected.")

for (x, y, w, h) in faces_rect:

    face_roi = gray[y:y+h, x:x+w]

    # Same preprocessing as training
    face_roi = cv.resize(face_roi, FACE_SIZE)

    label, confidence = face_recognizer.predict(face_roi)

    print(f"\nPredicted Label: {people[label]}")
    print(f"Confidence: {confidence:.2f}")
    print(f"ROI Shape: {face_roi.shape}")

    if confidence > UNKNOWN_THRESHOLD:
        name = "Unknown"
        color = (0, 0, 255)
    else:
        name = people[label]
        color = (0, 255, 0)

    cv.putText(img, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)

gray_display = rescaleFrame(gray)
img_display = rescaleFrame(img)

cv.imshow('Grayscale', gray_display)
cv.imshow('Recognition Result', img_display)

cv.waitKey(0)
cv.destroyAllWindows()