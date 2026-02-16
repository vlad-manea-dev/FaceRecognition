import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing images of known people
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(f'Found files: {myList}')

for cl in myList:
    if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        else:
            print(f"Could not read image: {cl}")

print(f'Known persons: {classNames}')

def findEncodings(images, classNames):
    encodeList = []
    for img, name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print(f"No face found in image: {name}")
    return encodeList

def markAttendance(name):
    csv_path = 'Attendance.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('Name,Time')

    with open(csv_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.strip().split(',')
            if entry:
                nameList.append(entry[0])
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
            print(f"Logged attendance for: {name} at {dtString}")
        else:
            # Optional: print only occasionally to avoid spam
            pass

encodeListKnown = findEncodings(images, classNames)
print('Encoding Complete')

# Try to open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Troubleshooting for macOS:")
    print("1. Go to System Settings > Privacy & Security > Camera.")
    print("2. Ensure your Terminal or IDE (e.g., PyCharm) is toggled ON.")
    print("3. If you have an iPhone, try turning off 'Continuity Camera' in Settings > General > AirPlay & Handoff.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam. If you have multiple cameras, try changing the index in cv2.VideoCapture(0).")
        break
    
    # Resize image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            distance = faceDis[matchIndex]

            if distance < 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = 'Unknown'
            
            # Print distance for debugging
            print(f"Recognized: {name} (Distance: {distance:.2f})")
        else:
            name = 'Unknown'

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
