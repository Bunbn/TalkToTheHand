# Import all libraries
import cv2
import numpy as np
import mediapipe as mp                         # Used for detecting & tracking hand landmarks
import time
from tensorflow.keras.models import load_model # Load pre-trained ASL model
from utils import extract_keypoints            # Custom function to extract 63 keypoints
from collections import deque, Counter         # Smooth predictions using history

# Load trained ASL classification model
model = load_model('model/asl_model.h5')

# Map class indicies to letters
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'YES', 27: 'NO', 28: 'THANK YOU', 29: 'HELLO', 30: 'PLEASE'
}
# added yes, no, thank you, hello, please

# Initialize a history buffer for smoothing
prediction_history = deque(maxlen=10)

# add new deque for keypoint history
keypoint_window = deque(maxlen=30)  
# store last 30 keypoint arrays

# Initialize a word output
word_buffer = ""    # String to build word
last_letter = None  # Tracks last letter added to avoid duplicates
last_added_time = 0 # Time stamp for last letter added
letter_delay = 1.0  # Seconds between letter additions

mp_hands = mp.solutions.hands           # Load the hands module
mp_drawing = mp.solutions.drawing_utils # Utility for drawing hand connections on screen

cap = cv2.VideoCapture(0) # Open the default webcam

# Main processing loop: Create a hand detector that tracks 1 hand
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():       # Continue while the camera is running
        ret, frame = cap.read() # Read one frame from the camera
        if not ret:             # If frame is read incorrectly, break
            break
        
        # Convert and process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert image to RGB (for MediaPipe)
        result = hands.process(image)                  # Run hand detection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert image to BGR (for OpenCV display)

        # Check if 1 hand is detected
        if result.multi_hand_landmarks:
            # Loop for 1 hand
            for hand_landmarks in result.multi_hand_landmarks:

                # update keypoint history and run prediction when full
                keypoints = extract_keypoints(hand_landmarks)
                keypoint_window.append(keypoints)

                # only predict when we have 30 frames
                if len(keypoint_window) == 30:
                    sequence_input = np.expand_dims(np.array(keypoint_window), axis=0)  # shape: (1, 30, 63)
                    prediction = model.predict(sequence_input)[0]                       # Run prediction
                    class_id = np.argmax(prediction)                                    # Get index of highest confidence class
                    prediction_history.append(class_id)                                 # Add prediciton to smoothing history
                # if we have not received 30 frames yet, do not predict
                else:
                    class_id = None # no class id is assigned


                # wrap prediction logic to only display labels when class_id is valid
                if class_id is not None:

                    # Get probability of predicted class
                    confidence = prediction[class_id]
                    if confidence > 0.6:
                        label = f"{label_map[class_id]} ({confidence * 100:.1f}%)"
                    else:
                        label = "Uncertain"
                    
                    # Get most frequent prediction in history
                    most_common_id, count = Counter(prediction_history).most_common(1)[0]

                    # Display only if majority consensus
                    if count > 5: 
                        predicted_letter = label_map[most_common_id]
                        confidence = prediction[most_common_id]

                        # Add letter if confidence is high and delay passed
                        if confidence > 0.9:
                            label = f"{predicted_letter} ({confidence * 100:.1f}%)"
                            current_time = time.time()
                            if last_letter != predicted_letter or (current_time - last_added_time) > letter_delay:
                                word_buffer += predicted_letter
                                last_letter = predicted_letter
                                last_added_time = current_time
                        
                        # Handle uncertainty
                        else:
                            label = "Uncertain"
                            last_letter = None  # Reset to allow re-entry later
                    
                    # Handle flickering
                    else:
                        label = "..."
                        last_letter = None  # Reset on instability

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Draw the predicted letter
                    cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 0, 0), 2, cv2.LINE_AA)
                    # Draw the current word
                    cv2.putText(image, f"Word: {word_buffer}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2, cv2.LINE_AA)


        # Show the result window
        cv2.imshow('ASL Recognition', image)

        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # Quit program
            break
        elif key == ord('d'): # Clear word
            word_buffer = ""
            last_letter = None
        elif key == ord('b'): # Remove last letter in word
            word_buffer = word_buffer[:-1]  

cap.release()           # Release the webcam
cv2.destroyAllWindows() # Close the OpeCV window