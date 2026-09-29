# Face Recognition Based Eye-Controlled Mouse

A hands-free mouse for Windows/Linux/macOS. The user is first verified by **face recognition**, then controls the cursor with **head movement, blinks, winks and mouth gestures** detected from a webcam. It's aimed at accessibility for people who can't use a physical mouse.

## How it works
1. **Register:** enter a name and capture face samples from the webcam (`Take Images`).
2. **Train:** face embeddings are computed with `face_recognition` and saved (`Train Images`).
3. **Log in:** the app confirms the webcam face matches the entered username.
4. **Control:** dlib's 68-point facial landmarks drive the cursor via PyAutoGUI.

### Gestures
| Gesture | Action |
|---|---|
| Open mouth (hold) | Toggle input mode on/off, and set the nose anchor point |
| Move head (nose away from anchor) | Move cursor left / right / up / down |
| Squint both eyes (hold) | Toggle scroll mode (head up/down then scrolls) |
| Wink left eye | Left click |
| Wink right eye | Right click |

Gestures are detected using the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** computed in `utils.py`.

## Tech stack
Python · OpenCV · dlib · face_recognition · imutils · PyAutoGUI · Tkinter

## Setup
```bash
git clone git@github.com:agvs03/Face-Recognition-Based-Eye-Controlled-Mouse.git
cd Face-Recognition-Based-Eye-Controlled-Mouse/Face_Recognition_Based_Eye_Control-Mouse_Cursor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> `dlib` needs CMake and a C++ compiler. On Windows, installing a prebuilt wheel (for example via `pip install dlib-bin`) is often easier.

Download the landmark model [`shape_predictor_68_face_landmarks.dat.bz2`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place the `.dat` file in `model/`.

## Run
```bash
python main.py
```

## Project structure
```
main.py            # Tkinter GUI: register, train, login, cursor control loop
extract_faces.py   # Builds face encodings from dataset/
utils.py           # EAR / MAR / direction helpers
model/             # Place the dlib landmark model here
dataset/           # Captured face images (git-ignored)
```
