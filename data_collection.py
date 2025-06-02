import cv2
import mediapipe as mp
import numpy as np
import os
from utils import extract_keypoints

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'data/'
LABEL = 'Z'  # Change per session
os.makedirs(DATA_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = extract_keypoints(hand_landmarks)
                np.save(os.path.join(DATA_PATH, f'{LABEL}_{count}.npy'), keypoints)
                count += 1

        cv2.imshow('Collecting Data', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()