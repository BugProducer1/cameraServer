import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response

app = Flask(__name__)

# --- AI MODEL SETUP ---
# 1. Load your trained .h5 or SavedModel
# model = tf.keras.models.load_model('my_garbage_model.h5')

# 2. Define your class names
# class_names = ['plastic', 'paper', 'metal']
# --------------------

def run_ai_on_image(image_bytes):
    print("Image received, running AI model...")

    # --- AI INFERENCE ---
    # 1. Decode the image
    # nparr = np.frombuffer(image_bytes, np.uint8)
    # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Pre-process the image
    # img_resized = cv2.resize(frame, (96, 96))
    # img_array = tf.keras.utils.img_to_array(img_resized)
    # img_batch = np.expand_dims(img_array, axis=0)

    # 3. Run prediction
    # prediction = model.predict(img_batch)
    # score = tf.nn.softmax(prediction[0])

    # 4. Get the result
    # label = class_names[np.argmax(score)]
    # confidence = 100 * np.max(score)

    # print(f"AI Result: {label} ({confidence:.2f}%)")

    # --- Placeholder for this example ---
    label = "plastic"
    # --------------------

    return label

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image_data = request.data

        result = run_ai_on_image(image_data)

        return Response(response=result, status=200, mimetype='text/plain')

    except Exception as e:
        print(f"Error processing image: {e}")
        return Response(response="Error", status=500, mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)