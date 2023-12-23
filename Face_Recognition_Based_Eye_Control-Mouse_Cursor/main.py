from tkinter import *
import tkinter as tk

from imutils import face_utils
from utils import *
import pyautogui as pag
import imutils
import dlib

import face_recognition
import pickle
import numpy as np
import cv2

from extract_faces import trainmodel

window = tk.Tk()
window.title("Face Recognition Based Eye Controlled Mouse Cursor")
window.geometry('1280x720')
window.configure(background='snow')

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

def clear():
    txt.delete(first=0, last=22)

def deleteScreen():
    screen.destroy()

def error_screen():

    global screen

    screen = tk.Tk()
    screen.geometry('300x100')
    screen.iconbitmap('AMS.ico')
    screen.title('Warning!!')
    screen.configure(background='snow')
    Label(screen,text='Name required!!!',fg='red',bg='white',font=('times', 16, ' bold ')).pack()
    Button(screen,text='OK',command=deleteScreen,fg="black"  ,bg="lawn green"  ,width=9  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold ')).place(x=90,y= 50)

def capctureimages():

    name = txt.get()

    if name == '':
        error_screen()
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            Name = txt.get()
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    cv2.imwrite("dataset/" + Name +"_"+ str(sampleNum) + ".jpg",gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 70:
                    break
            cam.release()
            cv2.destroyAllWindows()
            message = "Images Saved Successfully"
            Notification.configure(text=message, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=400)
        except FileExistsError as F:
            message = 'Failed'
            Notification.configure(text=message, bg="Red", width=21)
            Notification.place(x=450, y=400)

def trainimg():
    trainmodel()
    message = "Model Saved Successfully"
    Notification.configure(text=message, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=400)


def runapp():

    win = tk.Tk()
    win.iconbitmap('logo.ico')
    win.title("Mouse Control")
    win.geometry('880x420')
    win.configure(background='snow')

    def start_run():

        uname = un_entr.get()
        data = pickle.loads(open('face_enc', "rb").read())

        vid = cv2.VideoCapture(0)
        cam_w = 640
        cam_h = 480

        MOUTH_AR_THRESH = 0.6
        MOUTH_AR_CONSECUTIVE_FRAMES = 15
        EYE_AR_THRESH = 0.19
        EYE_AR_CONSECUTIVE_FRAMES = 15
        WINK_AR_DIFF_THRESH = 0.04
        WINK_CONSECUTIVE_FRAMES = 10

        MOUTH_COUNTER = 0
        EYE_COUNTER = 0
        WINK_COUNTER = 0
        INPUT_MODE = False
        SCROLL_MODE = False
        ANCHOR_POINT = (0, 0)
        YELLOW_COLOR = (0, 255, 255)
        RED_COLOR = (0, 0, 255)
        GREEN_COLOR = (0, 255, 0)
        BLUE_COLOR = (255, 0, 0)

        shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        while True:

            _, frame = vid.read()
            img=frame

            #========================================

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            predicted=""
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(data["encodings"], encodeFace)
                faceDis = face_recognition.face_distance(data["encodings"], encodeFace)
                matchIndex = np.argmin(faceDis)
                #print(matches, matchIndex)

                if matches[matchIndex]:
                    Id = data["names"][matchIndex]

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                    predicted = Id

                    #print(" Loop Predicted Username:", predicted)

            #print("input username:"+uname, " Predicted Username:"+predicted)

            if uname == predicted:

                #print("in if")
                frame = cv2.flip(frame, 1)
                frame = imutils.resize(frame, width=cam_w, height=cam_h)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.putText(img, str(Id), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                rects = detector(gray, 0)

                if len(rects) > 0:
                    rect = rects[0]
                else:
                    cv2.imshow("Frame", frame)
                    #print("in done retrive")
                    cv2.waitKey(1) & 0xFF
                    continue

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                mouth = shape[mStart:mEnd]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                nose = shape[nStart:nEnd]

                temp = leftEye
                leftEye = rightEye
                rightEye = temp

                mar = mouth_aspect_ratio(mouth)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                diff_ear = np.abs(leftEAR - rightEAR)

                nose_point = (nose[3, 0], nose[3, 1])

                mouthHull = cv2.convexHull(mouth)
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
                cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
                cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

                for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
                    cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

                if diff_ear > WINK_AR_DIFF_THRESH:

                    if leftEAR < rightEAR:
                        if leftEAR < EYE_AR_THRESH:
                            WINK_COUNTER += 1

                            if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                                pag.click(button='left')

                                WINK_COUNTER = 0

                    elif leftEAR > rightEAR:
                        if rightEAR < EYE_AR_THRESH:
                            WINK_COUNTER += 1

                            if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                                pag.click(button='right')

                                WINK_COUNTER = 0
                    else:
                        WINK_COUNTER = 0
                else:
                    if ear <= EYE_AR_THRESH:
                        EYE_COUNTER += 1

                        if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                            SCROLL_MODE = not SCROLL_MODE
                            EYE_COUNTER = 0
                    else:
                        EYE_COUNTER = 0
                        WINK_COUNTER = 0

                if mar > MOUTH_AR_THRESH:
                    MOUTH_COUNTER += 1

                    if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                        INPUT_MODE = not INPUT_MODE
                        MOUTH_COUNTER = 0
                        ANCHOR_POINT = nose_point

                else:
                    MOUTH_COUNTER = 0

                if INPUT_MODE:
                    cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                    x, y = ANCHOR_POINT
                    w, h = 60, 35
                    cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
                    cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

                    dir = direction(nose_point, ANCHOR_POINT, w, h)
                    cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
                    drag = 18
                    if dir == 'right':
                        pag.moveRel(drag, 0)
                    elif dir == 'left':
                        pag.moveRel(-drag, 0)
                    elif dir == 'up':
                        if SCROLL_MODE:
                            pag.scroll(40)
                        else:
                            pag.moveRel(0, -drag)
                    elif dir == 'down':
                        if SCROLL_MODE:
                            pag.scroll(-40)
                        else:
                            pag.moveRel(0, drag)

                if SCROLL_MODE:
                    cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

                cv2.putText(frame, str(predicted), (500, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

            else:
                cv2.putText(frame, "Invalid User", (500, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

        cv2.destroyAllWindows()
        vid.release()

    username = tk.Label(win, text="Enter username", width=15, height=2, fg="white", bg="blue2",font=('times', 15, ' bold '))
    username.place(x=30, y=50)

    un_entr = tk.Entry(win, width=20, bg="yellow", fg="red", font=('times', 23, ' bold '))
    un_entr.place(x=290, y=55)

    def clear():
        un_entr.delete(first=0, last=22)

    c0 = tk.Button(win, text="Clear", command=clear, fg="black", bg="deep pink", width=10, height=1,activebackground="Red", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    Login = tk.Button(win, text="LogIn", fg="black", bg="lime green", width=20,height=2,activebackground="Red",command=start_run, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.iconbitmap('logo.ico')

message = tk.Label(window, text="Face Recognition Based Eye Controlled Mouse Cursor", bg="cyan", fg="black", width=50,height=3, font=('times', 30, 'italic bold '))
message.place(x=80, y=20)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,height=3, font=('times', 17, 'bold'))

lbl = tk.Label(window, text="Enter Name", width=20, height=2, fg="black", bg="deep pink", font=('times', 15, ' bold '))
lbl.place(x=200, y=200)

txt = tk.Entry(window, validate="key", width=20, bg="yellow", fg="red", font=('times', 25, ' bold '))
txt.place(x=550, y=210)

clearButton = tk.Button(window, text="Clear",command=clear,fg="black"  ,bg="deep pink"  ,width=10  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=210)

takeImg = tk.Button(window, text="Take Images",command=capctureimages,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=500)

trainImg = tk.Button(window, text="Train Images",fg="black",command=trainimg,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=390, y=500)

runapp = tk.Button(window, text="Track User",fg="white",command=runapp,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
runapp.place(x=690, y=500)

window.mainloop()