import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sys

IMG_WIDTH = 96
IMG_HEIGHT = 96
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 16
DATA_DIR = "dataset"

print(f"Loading images from: {DATA_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Found classes: {class_names}")

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(class_names), activation='softmax') # Use softmax for multi-class
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\nStarting model training...")
# You can adjust epochs based on previous results, 15 seemed okay
model.fit(train_ds, validation_data=val_ds, epochs=15)

print("\nConverting model to TensorFlow Lite (.tflite) WITHOUT optimization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- OPTIMIZATION REMOVED ---
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # This line is commented out/removed
# ----------------------------

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Success! Saved UNOPTIMIZED model as 'model.tflite' ({len(tflite_model)} bytes)")
print("Phase 1 is complete. Add model.tflite to your GitHub repo.")

# Update app.py class_names (optional helper)
print("\n--- Update app.py ---")
print("Make sure the class_names list in your app.py matches this order:")
print(f"class_names = {class_names}")
print("---------------------\n")