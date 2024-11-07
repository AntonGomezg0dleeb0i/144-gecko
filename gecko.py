import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
def load_images(data_dir, img_size):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Get class names from subfolders
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            # Only process valid image files
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skipping non-image file: {img_file}")
                continue
            try:
                # Load the image, resize, and convert to grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names

# Set parameters
train_dir = "Train"  # Path to the training dataset
test_dir = "Test"    # Path to the testing dataset
img_size = 64        # Resize images to 64x64 pixels

# Load training and testing data
X_train, y_train, class_names = load_images(train_dir, img_size)
X_test, y_test, _ = load_images(test_dir, img_size)  # Use the same class names

# Normalize images
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
X_test = X_test / 255.0

# Add a channel dimension for grayscale images
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 2. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
    layers.RandomRotation(0.05),      # Rotate images by ±10%
    layers.RandomZoom(0.05)           # Randomly zoom into the image
])

# 3. Build the Model with Dropout and Augmentation
model = models.Sequential([
    data_augmentation,  # Apply data augmentation to inputs
    layers.Input(shape=(img_size, img_size, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  # Drop 25% of neurons
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Drop 25% of neurons
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  # Drop 50% of neurons
    layers.Dense(len(class_names), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

# 6. Visualize Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 7. Predict on New Images
def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    prediction = model.predict(img)
    predicted_label = class_names[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_label}")
    return predicted_label

# Example prediction
sample_img = "Test/Dalmatian/image1.jpg"  # Replace with your test image path
predict_image(sample_img)

