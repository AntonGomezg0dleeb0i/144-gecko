import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 1. Load and Preprocess Data
def load_images(data_dir, img_size):
    images = []
    labels = []
    traits = ["Dalmatian", "Pinstripe"]  # Define traits for multi-label classification

    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Extract labels based on file names or directory structure
        label_vector = [1 if trait.lower() in img_file.lower() else 0 for trait in traits]
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label_vector)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, traits

# Set parameters
train_dir = "Train"  # Path to the training dataset
test_dir = "Test"    # Path to the testing dataset
img_size = 396       # Resize images to 396 pixels

# Load training and testing data
X_train, y_train, traits = load_images(train_dir, img_size)
X_test, y_test, _ = load_images(test_dir, img_size)  # Use the same traits

# Normalize images
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
X_test = X_test / 255.0

# Add a channel dimension for grayscale images
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 2. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
    layers.RandomRotation(0.05),      # Rotate images by ±5%
    layers.RandomZoom(0.05)           # Randomly zoom into the image
])

# 3. Build the Model with Dropout and Augmentation
model = models.Sequential([
    data_augmentation,  # Apply data augmentation to inputs
    layers.Input(shape=(img_size, img_size, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  # Drop neurons during training
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Drop neurons during training
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  # Drop neurons during training
    layers.Dense(len(traits), activation='sigmoid')  # Multi-label output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

# Detailed metrics
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Threshold predictions
print(classification_report(y_test, y_pred, target_names=traits))

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
    
    prediction = model.predict(img)[0]
    predicted_traits = [traits[i] for i, p in enumerate(prediction) if p > 0.5]
    
    print(f"Predicted Traits: {', '.join(predicted_traits) if predicted_traits else 'None'}")
    return predicted_traits

# Example prediction
sample_img = "Test/Dalmatian/image1.jpg"  # Replace with your test image path
predict_image(sample_img)
