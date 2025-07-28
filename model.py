import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load your dataset folder
data_dir = "asl_dataset"  # Make sure this folder contains A-Z folders with images
classes = sorted(os.listdir(data_dir))  # ['A', 'B', ..., 'Z']
X = []
y = []

for i, letter in enumerate(classes):
    folder = os.path.join(data_dir, letter)
    if os.path.isdir(folder):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            X.append(img)
            y.append(i)

X = np.array(X).reshape(-1, 64, 64, 1) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')  # output 26 letters
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("sign_language_model.h5")
