# TalkToTheHand — Real-Time American Sign Language Recognition

**Visionary Bytes Baddies**  
Alain Kanadjian, Emily Nicoletta, Katelynn Hoang

---

## Overview

TalkToTheHand is a computer vision system designed to recognize American Sign Language (ASL) gestures from webcam video input in real-time. The system uses hand keypoint detection combined with deep learning to translate hand signs into letters or words displayed on screen.

---

## Features

- Real-time hand gesture detection using webcam input
- Hand keypoint extraction using [MediaPipe](https://mediapipe.dev/)
- https://research.google/blog/on-device-real-time-hand-tracking-with-mediapipe/ 
- ASL sign classification with TensorFlow Lite models
- On-screen output translation of recognized signs
- Data collection script for capturing training data

---

## Requirements
- Run requirements.txt
- See `requirements.txt` for full Python dependencies
  
---

## Improvements
- Accuracy/certainty percentage on screen
- If multiple hands, letters moving to each specific hand
- Change model training method and check accuracy for each letter by putting in photos 

---

## Setup

1. Setup VSCODE with your github account
2. Create a folder for this and then clone the repository (git clone https://github.com/Bunbn/TalkToTheHand.git
   cd TalkToTheHand
3. pip install -r requirements.txt
4. LABEL = 'C'  # Change per session Adjust this for the sign language letter you do
5. run 'python data_collection.py' to train, dont run for too long
6. Itll create npy based on the label letter, so move ur hand around doing the specific label letter in sign language.
7. This will save in the data folder
8. Then run python train.py to make the model
9. Then python real_time_actions.py to get the camera up for recognizing
