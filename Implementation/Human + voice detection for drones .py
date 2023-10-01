import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import sounddevice as sd

## Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def Detector(frame):
    ## Using Sliding window concept
    rects, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    c = 1
    for x, y, w, h in pick:
        cv2.rectangle(frame, (x, y), (w, h), (139, 34, 104), 2)
        cv2.rectangle(frame, (x, y - 20), (w, y), (139, 34, 104), -1)
        cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        c += 1

    cv2.putText(frame, f'Total Persons : {c - 1}', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('output', frame)
    return frame

def sound_detection():
    duration = 5  # Duration in seconds to check for continuous sound
    sample_rate = 44100  # Sample rate (you can adjust this)

    print("Listening for continuous sound...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    # Check if sound levels exceed a threshold (you can adjust this)
    threshold = 0.05  # Adjust as needed
    if np.max(np.abs(audio_data)) > threshold:
        print("Person is continuously screaming for more than 5 seconds!")

cap = cv2.VideoCapture(0)  # Use camera index 0 for the default camera

while True:
    ret, frame = cap.read()
    frame = Detector(frame)
    
    # Check for sound detection
    sound_detection()
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()
