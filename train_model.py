import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# Define gesture labels
gestures = ['Rock', 'Paper', 'Scissors']

# Define paths (adjust if needed)
train_path = os.path.join(os.path.expanduser('~'),'OneDrive', 'Desktop', 'CV_Project', 'images', 'Train')
test_path = os.path.join(os.path.expanduser('~'), 'OneDrive','Desktop', 'CV_Project', 'images', 'Test')

def load_images_from_folder(folder_path):
    data = []
    labels = []

    # Loop through each gesture
    for label, gesture in enumerate(gestures):
        gesture_folder = os.path.join(folder_path, gesture)
        if not os.path.exists(gesture_folder):
            print(f"Folder not found: {gesture_folder}")
            continue

        # Process each image in the folder
        for img_name in os.listdir(gesture_folder):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only accept image files
                continue

            img_path = os.path.join(gesture_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {img_path}")
                continue

            try:
                # Resize image to (64, 64)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fixed the bug here
                img = cv2.resize(img, (64, 64))
                img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    data = np.array(data)
    labels = np.array(labels)

    print(f"Loaded {len(data)} images from {folder_path}")
    return data, labels

# Load training and test data
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(test_path)

# Check the shapes of the data and labels
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Ensure data is correctly shaped for training
assert X_train.ndim == 4 and X_train.shape[1:] == (64, 64, 3), "X_train has incorrect shape"
assert y_train.ndim == 1, "y_train should be a 1D array"
assert X_test.ndim == 4 and X_test.shape[1:] == (64, 64, 3), "X_test has incorrect shape"
assert y_test.ndim == 1, "y_test should be a 1D array"

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Added dropout to prevent overfitting
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (increased epochs and added validation split)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('rps_model.keras')
print("Model saved as rps_model.keras")
