import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sys

IMG_WIDTH = 96
IMG_HEIGHT = 96
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 16 # Batch size is 16, but you only have 4 images.
DATA_DIR = "dataset"
EPOCHS = 50 # Reduced epochs, no early stopping

print(f"Loading images from: {DATA_DIR}")

# --- Load Datasets ---
# NOTE: We have REMOVED the validation_split.
# With so few images, you cannot split the data.
# This will fix your ValueError.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    # No validation_split or subset
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found classes: {class_names}")

# --- Configure for Performance ---
AUTOTUNE = tf.data.AUTOTUNE
# You only have 4 images, but this is good practice
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation Layer ---
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)

# --- Build a More Robust Model ---
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, name="rescaling"),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax', name="output")
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- Early Stopping REMOVED ---
# We have no validation data, so we cannot use EarlyStopping.

print("\nStarting model training...")
history = model.fit(
    train_ds, 
    # validation_data=val_ds, # REMOVED
    epochs=EPOCHS
    # callbacks=[early_stopper] # REMOVED
)

print("\nTraining complete.")
# We cannot evaluate on a validation set, because we don't have one.


print("\nConverting model to TensorFlow Lite (.tflite) WITHOUT optimization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Success! Saved UNOPTIMIZED model as 'model.tflite' ({len(tflite_model)} bytes)")
print("Phase 1 is complete. Add model.tflite to your GitHub repo.")

print("\n--- IMPORTANT! ---")
print("Make sure the class_names list in your app.py matches this EXACT order:")
print(f"class_names = {class_names}")
print("---------------------\n")