import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import extract_keypoints
from collections import deque, Counter

model = load_model('model/asl_model.h5')
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# Initialize a history buffer
prediction_history = deque(maxlen=10)

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
                prediction_history.append(class_id)

                # Add confidence
                confidence = prediction[class_id]  # Get certainty of top prediction
                if confidence > 0.6:
                    label = f"{label_map[class_id]} ({confidence * 100:.1f}%)"
                else:
                    label = "Uncertain"
                
                # Get most frequent prediction in history
                most_common_id, count = Counter(prediction_history).most_common(1)[0]

                # Display only if majority consensus
                if count > 5:
                    label = f"{label_map[most_common_id]} ({prediction[most_common_id] * 100:.1f}%)"
                else:
                    label = "..."

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('ASL Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()