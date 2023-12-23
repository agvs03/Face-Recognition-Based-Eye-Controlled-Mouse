import face_recognition
import pickle
import numpy as np
import cv2

cascPathface = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())

cam = cv2.VideoCapture(0)

while (True):

    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(data["encodings"], encodeFace)
        faceDis = face_recognition.face_distance(data["encodings"], encodeFace)
        matchIndex = np.argmin(faceDis)
        print(matches, matchIndex)

        if matches[matchIndex]:
            Id = data["names"][matchIndex]

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(Id), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            accountnumber = Id

    cv2.imshow('Webcam', img)
    # wait for 100 miliseconds
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()