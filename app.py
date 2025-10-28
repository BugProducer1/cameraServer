import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, Response

app = Flask(__name__)

# --- AI MODEL SETUP ---
MODEL_PATH = "model.tflite" # Your unoptimized model file
interpreter = None
input_details = None
output_details = None
IMG_HEIGHT = 96 # Should match the training script (train.py)
IMG_WIDTH = 96 # Should match the training script (train.py)

# !! IMPORTANT: Make sure this list EXACTLY matches the order from train.py !!
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

try:
    print(f"Loading TFLite model from: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model loaded successfully. Expecting input shape: (1, {IMG_HEIGHT}, {IMG_WIDTH}, 3)")

except Exception as e:
    print(f"ERROR: Failed to load model or allocate tensors: {e}")
    # Keep interpreter as None to indicate failure

# --------------------

def run_ai_on_image(image_bytes):
    if interpreter is None:
        print("ERROR: Interpreter not available.")
        return "Error: Model not loaded"

    print("Image received, running AI model...")

    try:
        # --- AI INFERENCE ---
        # 1. Decode the image (OpenCV reads as BGR by default)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("ERROR: Failed to decode image.")
            return "Error: Bad image data"

        # ** Convert color from BGR to RGB **
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Pre-process the image (use the RGB frame)
        img_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT)) # Use frame_rgb
        img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
        # Normalize: Convert pixel values from 0-255 to 0.0-1.0
        img_batch = img_batch / 255.0

        # 3. Set tensor, invoke (run inference), and get results
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # 4. Get the result
        scores = prediction[0]
        predicted_index = np.argmax(scores)
        # ADD THIS LINE TO LOG SCORES (optional, but helpful for debugging):
        print(f"Raw Scores: {scores}")
        label = class_names[predicted_index]
        confidence = 100 * scores[predicted_index]

        print(f"AI Result: {label} ({confidence:.2f}%)")
        return label

    except Exception as e:
        print(f"ERROR during inference: {e}")
        return "Error: Inference failed"


@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        try:
            image_data = request.data
            if not image_data:
                return Response(response="Error: No image data received", status=400, mimetype='text/plain')

            result = run_ai_on_image(image_data)
            return Response(response=result, status=200, mimetype='text/plain')

        except Exception as e:
            print(f"Error processing upload request: {e}")
            return Response(response="Server Error", status=500, mimetype='text/plain')
    else:
        return Response(response="Method Not Allowed", status=405, mimetype='text/plain')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    # Make sure debug=False for production on Render
    app.run(host='0.0.0.0', port=port, debug=False)