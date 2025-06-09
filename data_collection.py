# Import all libraries
import cv2
import mediapipe as mp              # Used for detecting & tracking hand landmarks
import numpy as np
import os
from utils import extract_keypoints # Convert hand landmarks to a 63-length array

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'data/'                   # Directory to store .npy files
LABEL = 'A'                           # Change per session (e.g. 'B', 'C', 'D', etc.)
os.makedirs(DATA_PATH, exist_ok=True) # Change directory if it doesnt already exist

cap = cv2.VideoCapture(0) # Open the default webcam

# Setup MediaPipe Hand Detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    count = 0 # Counter for saved frames
    
    # Main loop for capturing data
    while cap.isOpened():       # While the webcam is running, keep looping
        ret, frame = cap.read() # Read the frame from the webcam
        if not ret:             # If the camera read fails, break
            break
        
        # Convert and process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert image to RGB (for MediaPipe)
        result = hands.process(image)                  # Run hand detection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert image to BGR (for OpenCV display)

        # If hand is detected, save the data
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = extract_keypoints(hand_landmarks)                               # Convert to a flat array
                np.save(os.path.join(DATA_PATH, f'{LABEL}_{count}.npy'), keypoints)         # Save to file
                count += 1                                                                  # Increment saved file count

        # Show frame with drawn landmarks
        cv2.imshow('Collecting Data', image)
        # Wait for keypress
        if cv2.waitKey(1) & 0xFF == ord('q'): # Quit program
            break

cap.release()           # Release the webcam
cv2.destroyAllWindows() # Close the OpenCV window