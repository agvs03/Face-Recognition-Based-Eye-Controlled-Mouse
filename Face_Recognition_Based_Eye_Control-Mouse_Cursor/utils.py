"""Facial-landmark helpers used by main.py for eye/mouth gestures and cursor direction."""
import numpy as np


def eye_aspect_ratio(eye):
    """Eye Aspect Ratio (Soukupova & Cech, 2016) from the 6 eye landmarks.

    Drops toward 0 when the eye closes, so it is used to detect blinks and winks.
    """
    # Vertical distances between the upper and lower eyelid landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Horizontal distance between the eye corners
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """Mouth Aspect Ratio from the 20 mouth landmarks; rises when the mouth opens."""
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])
    return (A + B + C) / (2.0 * D)


def direction(nose_point, anchor_point, w, h, multiple=1):
    """Return 'left', 'right', 'up', 'down' or 'none' depending on where the nose
    sits relative to a w x h box centred on the anchor point."""
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'

    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'

    return 'none'
