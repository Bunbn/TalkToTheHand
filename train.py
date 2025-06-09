# Import all libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout   # Dense: fully connected layers; Dropout: regularization
from tensorflow.keras.utils import to_categorical    # Converts class labels into one-hot encoding

DATA_PATH = 'data/' # Folder contatining .npy files (keypoint data)
X, y = [], []       # X = input data, y = class labels

# Convert letters to class IDs
label_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25
}  # Expand this

# Loop through each file in the data folder
for file in os.listdir(DATA_PATH):
    key = file.split('_')[0]                      # Extract letter from filename
    data = np.load(os.path.join(DATA_PATH, file)) # Load the .npy file contents
    X.append(data)                                # Add keypoint data to input list
    y.append(label_map[key])                      # Convert letter to number and store as label

X = np.array(X)           # Convert input list to NumPy array
y = np.array(y)           # Convert label list to Numpy array
y_cat = to_categorical(y) # One-hot encode labels

# Train/test split: 80% train, 20% test
# Added stratify
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat)

# Build the neural network
model = Sequential([
    # First hidden layer
    Dense(256, activation='relu', input_shape=(63,)), # 63 inputs/keypoints
    Dropout(0.4),                                     # Drop 40% of nodes during training to prevent overfitting
    # Second hidden layer
    Dense(128, activation='relu'),                    
    Dropout(0.3),                                     # Drop 30% of nodes during training to prevent overfitting
    # Third hidden layer
    Dense(64, activation='relu'),
    # Output layer 
    Dense(len(label_map), activation='softmax')       # 26 neurons (A-Z): softmax for classification
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', # For multiclass classification
              metrics=['accuracy'])            # Track accuracy during training

# Train the model
model.fit(X_train, y_train,                 # Training data
          epochs=100,                       # Train for 100 passes over the dataset
          validation_data=(X_test, y_test)) # Evaluate on test set after each epoch

# Save the trained model to a file
model.save('model/asl_model.h5')