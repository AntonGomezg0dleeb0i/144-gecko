import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
import matplotlib.pyplot as plt
import ast
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# 1. Load and Process Trait Data
def load_and_process_traits(csv_path):
    df = pd.read_csv(csv_path)

    # Convert Tag column to list
    df["Tag"] = df["Tag"].apply(ast.literal_eval)  # Convert stringified lists to actual lists

    # Normalization map for simplifying traits (only patterns remain, remove color-related tags)
    normalization_map = {
        "Super Dalmatian": "Dalmatian",
        "Ink Spot": "Dalmatian",
        "Oil Spot": "Dalmatian",
        "Cluster Spots": "Dalmatian",
        "Extreme Harlequin": "Harlequin",
        "Tri-color": "Harlequin",
        "Halloween": "Harlequin",
        "Pos Tri-color": "Harlequin",
        "Pos Harlequin": "Flame",
        "Chevron": "Flame",
        "White Out": "White Wall",
        "Partial Pinstripe": "Pinstripe",
        "Pin-dashed": "Pinstripe",
        "Super Stripe": "Pinstripe",
        "Empty Back": "Pinstripe",
        "Red Spot": "Dalmatian",
        "Tiger": "Brindle",
        "Patternless": "Phantom",
        "Bi-color": "Phantom",
        "Snowflake": "Portholes",
        "Bold Stripe Tigers": "Brindle"
    }

    # Define color-related tags to remove
    color_tags = {"Brown", "Buckskin", "Paradox", "Cold Fusion", "Cappuccino", "Blonde", "Axanthic", "Tipped Crests", "Other Trait", "White Tip", "Pos Peppered",
                  "Soft Scale", "Orange Tip", "Pos Crowned", "Hybrid", "Tipped Crest", "Black", "Het Super Stripe", "50% Het Axanthic",
                  "Het Axanthic", "Pos Het Super Stripe", "Pos Het Axanthic", "Pos Brindle", "Normal", "Blushing", "Pos Cream",
                  "Pos Solid Back", "Lavender", "Pos Fringing", "Furred", "Pos Furred", "Lavender Albino", "Dinker", "Super Soft Scale",
                  "66% Het Axanthic", "Het Phantom", "White", "Pos Orange", "Pos Black", "Pos Dark", "Pos Lavender", "Pos Red", 
                  "Pos Portholes", "Pos Partial Pinstripe", "Pos Drippy", "Copper", "Red Base", "Dark", "Tailless", "Crowned", 
                  "Red", "Pos Quad-stripe", "Yellow", "Pos Dalmatian", "Pos Flame", "Cream", "Pink", "Orange", "Tangerine", 
                  "Mocha", "Peppered", "Pos Het Phantom", "Pos Halloween", "Creamsicle", "Pos Ink Spot", "Pos Extreme Harlequin", 
                  "Pos Olive", "Olive", "Pos Super Dalmatian"}

    def normalize_and_filter_tags(tags):
        # Normalize patterns and filter out color-related tags
        normalized_tags = [normalization_map.get(tag, tag) for tag in tags]
        return list(set(tag for tag in normalized_tags if tag not in color_tags))

    # Normalize and filter the tags in the Tag column
    df["Tag"] = df["Tag"].apply(normalize_and_filter_tags)

    # Flatten all tags to get unique traits
    all_tags = set(tag for tags in df["Tag"] for tag in tags)

    # Check tag distribution to validate balance
    print("Tag distribution after normalization and filtering:")
    print(pd.Series([tag for tags in df["Tag"] for tag in tags]).value_counts())

    # Generate one-hot encoded columns for all tags
    one_hot = pd.DataFrame(
        [{tag: 1 if tag in tags else 0 for tag in all_tags} for tags in df["Tag"]],
        index=df.index
    )
    df = pd.concat([df, one_hot], axis=1)

    return df

# Load traits data
csv_path = "trait-image.csv"
df = load_and_process_traits(csv_path)

# Split into training and testing datasets and reset indices
train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)  # 80% training
test_df = df.drop(train_df.index).reset_index(drop=True)

# 2. Load and Preprocess Images
def load_images_from_dataframe(df, image_dir, img_size):
    images = []
    labels = []
    valid_indices = []  # To track valid rows in the DataFrame

    columns_to_use = df.columns.difference(["Image Name", "Tag"])  # Use only pattern traits

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row["Image Name"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load image in color
            if img is None:  # Handle unreadable images
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            try:
                img = cv2.resize(img, (img_size, img_size))  # Resize to specified dimensions
                img = preprocess_input(img)  # Apply ResNet preprocessing
                images.append(img)
                labels.append(row[columns_to_use].values.astype(np.float32))  # Convert labels to float
                valid_indices.append(idx)  # Record valid indices
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
        else:
            print(f"Warning: Image file {img_path} does not exist.")

    images = np.array(images)
    labels = np.array(labels)

    # Filter the DataFrame to include only valid indices
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)

    return images, labels, filtered_df

# Load images and labels
image_dir = "images"
img_size = 224  # ResNet50 input size
X_train, y_train, train_df = load_images_from_dataframe(train_df, image_dir, img_size)
X_test, y_test, test_df = load_images_from_dataframe(test_df, image_dir, img_size)

# Normalize labels
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 3. Augment the Data
def augment_image(image):
    # Apply selected augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Adjust contrast
    return image

def augment_dataset(images, labels, augmentation_factor=2):
    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        for _ in range(augmentation_factor):
            augmented_images.append(augment_image(images[i]))
            augmented_labels.append(labels[i])
    return np.array(augmented_images), np.array(augmented_labels)

def augment_underrepresented_data(images, labels, df, num_classes, augmentation_factor=4):
    tag_counts = df.iloc[:, -num_classes:].sum(axis=0)  # Count tags
    underrepresented_tags = tag_counts[tag_counts < 400].index  # Define threshold for underrepresentation
    print(f"Underrepresented tags: {underrepresented_tags}")  # Debugging info

    # Get indices for underrepresented tags
    underrepresented_indices = df[df[underrepresented_tags].sum(axis=1) > 0].index.to_numpy()  # Ensure NumPy array
    underrepresented_images = images[underrepresented_indices]
    underrepresented_labels = labels[underrepresented_indices]
    
    # Augment data
    aug_images, aug_labels = augment_dataset(underrepresented_images, underrepresented_labels, augmentation_factor)
    return np.concatenate((images, aug_images)), np.concatenate((labels, aug_labels))

# Augment underrepresented training data
num_classes = y_train.shape[1]  # Number of traits to predict
X_train_aug, y_train_aug = augment_underrepresented_data(X_train, y_train, train_df, num_classes, augmentation_factor=8)

# Shuffle augmented data
X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

# 4. Build the Model
def build_resnet_model(img_size, num_classes):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # Gradually unfreeze deeper layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all but the last 10 layers
        layer.trainable = False

    # Add new layers on top for classification
    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=True)  # Pass images through ResNet base
    x = GlobalAveragePooling2D()(x)  # Pooling to reduce dimensions
    x = Dense(256, activation="relu")(x)  # Add a dense layer for learning features
    x = Dense(num_classes, activation="sigmoid")(x)  # Sigmoid for multi-label classification

    # Create the model
    model = Model(inputs, x)

    # Compile the model
    model.compile(optimizer="adam",  # Default learning rate
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

# Build the model
model = build_resnet_model(img_size, num_classes)

# 5. Define Learning Rate Warm-Up
def warmup_schedule(epoch):
    warmup_epochs = 5  # Number of epochs for warm-up
    initial_lr = 1e-6  # Starting learning rate
    target_lr = 1e-3   # Target learning rate

    if epoch < warmup_epochs:
        return initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
    return target_lr

warmup_scheduler = LearningRateScheduler(warmup_schedule)

# 6. Define Cyclical Learning Rate (CLR)
#from tensorflow.keras.optimizers.schedules import CyclicalLearningRate

#clr = CyclicalLearningRate(
    #initial_learning_rate=1e-6,
    #maximal_learning_rate=1e-3,
    #scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    #step_size=2000  # Number of steps per cycle
#)

# Update optimizer with CLR
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=clr),
              #loss="binary_crossentropy",
              #metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1
)

# 5. Train the Model
history_aug = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=32,
    callbacks=[lr_scheduler, warmup_scheduler]
)

# 6. Evaluate the Model
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")

# 7. Visualize Results
plt.plot(history_aug.history['accuracy'], label='Training Accuracy')
plt.plot(history_aug.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






