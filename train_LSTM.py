# Import all libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM    # Dense: fully connected layers; Dropout: regularization; LSTM: long short-term memory
from tensorflow.keras.utils import to_categorical           # Converts class labels into one-hot encoding

DATA_PATH = 'data/' # Folder contatining .npy files (keypoint data)
X, y = [], []       # X = input data, y = class labels

# Convert letters and word signs to class IDs
label_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 'YES': 26, 'NO': 27, 'THANK_YOU': 28, 'HELLO': 29, 'PLEASE': 30
}
# added yes, no, thank you, hello, please

# Loop through each file in the data folder 
for file in os.listdir(DATA_PATH):
    # load sequences from .npz files
    if file.endswith('.npz'):
        path = os.path.join(DATA_PATH, file)
        data = np.load(path)           # Load the .npz file contents
        sequences = data['sequences']  
        # shape: (n_samples, 30, 63)
        labels = data['labels']        
        # shape: (n_samples,)
        
        X.extend(sequences) 
        y.extend([label_map[str(l)] for l in labels])  
        # convert 'A' to 0, etc

# convert to numpy arrays
X = np.array(X)                 # Add keypoint data to input list
# shape: (total_samples, 30, 63)
y = to_categorical(np.array(y)) # Convert label list to Numpy array, One-hot encode labels
# shape: (total_samples, 26)

# Train/test split: 80% train, 20% test
# Added stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 63)), # 128 units, 30 sequence frames,63 inputs/keypoints
    Dropout(0.3),                                          # Drop 30% of nodes during training to prevent overfitting
    LSTM(64),                                               # 64 units
    Dropout(0.3),                                          # Drop 30% of nodes during training to prevent overfitting
    Dense(64, activation='relu'),                          # Output layer
    Dense(len(label_map), activation='softmax')            # 31 neurons (A-Z + words): softmax for classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # For multiclass classification
    metrics=['accuracy']             # Track accuracy during training
)

# Train model
history = model.fit(
    X_train, y_train,                # For multiclass classification
    epochs=100,                      # Train for 100 passes over the dataset
    validation_data=(X_test, y_test) # Evaluate on test set after each epoch
)

# Save the trained model to a file
os.makedirs('model', exist_ok=True)
model.save('model/asl_model.h5')


## plot model accuracy and loss
# import libraries
from matplotlib import pyplot as plt

# set figure size
plt.figure(figsize=(12, 5))

# first subplot is model accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('LSTM Model Accuracy')

# second plot is model loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('LSTM Model Loss')

# show figure
plt.show()