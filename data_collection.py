# Import all libraries
import cv2
import mediapipe as mp              # Used for detecting & tracking hand landmarks
import numpy as np
import os
from utils import extract_keypoints # Convert hand landmarks to a 63-length array

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'data/'                   # Directory to store .npz file
LABEL = 'PLEASE'                      # Change per session (e.g. 'B', 'C', 'D', etc.)
os.makedirs(DATA_PATH, exist_ok=True) # Change directory if it doesnt already exist

cap = cv2.VideoCapture(0) # Open the default webcam

# Initialize lists to hold data
SEQUENCE_LENGTH = 30
# set sequence length to 30 - represents 30 frames
sequence = []
sequences = []
labels = []

# Setup MediaPipe Hand Detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    
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
                keypoints = extract_keypoints(hand_landmarks) # Convert to a flat array
                sequence.append(keypoints)                    # append frame to a sequence

                # if the length of a sequence reaches 30 frames, append to sequences
                if len(sequence) == SEQUENCE_LENGTH:          
                    sequences.append(sequence) # sequences is multiple 30 frames long sequences for a specific sign
                    labels.append(LABEL)       # append label for specific sign
                    sequence = []              # clear for next sequence

        # Show frame with drawn landmarks
        cv2.imshow('Collecting Data', image)
        # Wait for keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()           # Release the webcam
cv2.destroyAllWindows() # Close the OpenCV window


# save all collected sequences to a .npz file - file that stores multiple .npy files
X = np.array(sequences)  
# shape: (num_sequences, 30, 63)
y = np.array(labels)     
# shape: (num_sequences,)
np.savez_compressed(os.path.join(DATA_PATH, f'{LABEL}_sequences.npz'), sequences=X, labels=y)

# print how many sequences (how many data point sets) were saved for that hand sign 
print(f"[INFO] Saved {len(X)} sequences to {LABEL}_sequences.npz")