import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import extract_keypoints

model = load_model('model/asl_model.h5')
label_map = {0: 'A', 1: 'B', 2: 'C'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = extract_keypoints(hand_landmarks)
                prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
                class_id = np.argmax(prediction)
                label = label_map[class_id]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('ASL Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()