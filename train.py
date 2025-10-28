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
EPOCHS = 100 # We'll use EarlyStopping, so this is just a max

print(f"Loading images from: {DATA_DIR}")

# --- Load Datasets ---
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
num_classes = len(class_names)
print(f"Found classes: {class_names}")

# --- Configure for Performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation Layer ---
# This layer will run on-GPU and only during training
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
    # Start with data augmentation
    data_augmentation,
    
    # Add the Rescaling layer. This is CRITICAL.
    # It makes the model expect 0-255 input, which fixes our bug.
    layers.Rescaling(1./255, name="rescaling"),
    
    # Block 1
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Block 2
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Block 3
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Add dropout to prevent overfitting
    layers.Dense(num_classes, activation='softmax', name="output") # Use softmax
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- Add Early Stopping ---
# This stops training when the validation accuracy stops improving.
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=10,             # Stop after 10 epochs of no improvement
    verbose=1,
    restore_best_weights=True # Keep the best model, not the last one
)

print("\nStarting model training...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=[early_stopper] # Add the callback here
)

print("\nTraining complete. Model accuracy on validation set:")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")


print("\nConverting model to TensorFlow Lite (.tflite) WITHOUT optimization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- OPTIMIZATION REMOVED (as in your original script) ---
# This is fine for now. It keeps the model as float32.
# converter.optimizations = [tf.lite.Optimize.DEFAULT] 
# --------------------------------------------------------

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Success! Saved UNOPTIMIZED model as 'model.tflite' ({len(tflite_model)} bytes)")
print("Phase 1 is complete. Add model.tflite to your GitHub repo.")

# Update app.py class_names (optional helper)
print("\n--- IMPORTANT! ---")
print("Make sure the class_names list in your app.py matches this EXACT order:")
print(f"class_names = {class_names}")
print("---------------------\n")