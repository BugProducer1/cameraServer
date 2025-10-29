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
EPOCHS = 50

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
num_classes = len(class_names)
print(f"Found classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)

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

early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

print("\nStarting model training...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=[early_stopper]
)

print("\nTraining complete. Model accuracy on validation set:")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")


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