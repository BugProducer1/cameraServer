import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, Response

app = Flask(__name__)

MODEL_PATH = "model.tflite"
interpreter = None
input_details = None
output_details = None
IMG_HEIGHT = 96
IMG_WIDTH = 96

# --- IMPORTANT ---
# Make sure this order matches the output from train.py
# (e.g., "Found classes: ['metal', 'plastic']")
class_names = ['metal', 'plastic']

try:
    print(f"Loading TFLite model from: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model loaded successfully. Expecting input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")

except Exception as e:
    print(f"ERROR: Failed to load model or allocate tensors: {e}")

def run_ai_on_image(image_bytes):
    if interpreter is None:
        print("ERROR: Interpreter not available.")
        return "Error: Model not loaded"

    print("Image received, running AI model...")

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)

        # --- START OF BUG FIX ---
        
        # 1. Decode the image using the correct flag (IMREAD_COLOR)
        #    This loads the image as BGR by default.
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("ERROR: Failed to decode image.")
            return "Error: Bad image data"

        # 2. Convert the image from BGR (OpenCV default) to RGB (TensorFlow default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- END OF BUG FIX ---

        img_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_batch = np.expand_dims(img_resized, axis=0)

        # Normalize or cast type based on model input requirements
        if input_details[0]['dtype'] == np.float32:
            img_batch = img_batch.astype(np.float32) 
        else:
            img_batch = img_batch.astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        scores = prediction[0]
        predicted_index = np.argmax(scores)
        
        print(f"Raw Scores: {scores}")
        
        label = class_names[predicted_index]
        confidence = 100 * scores[predicted_index] # Note: This is not a true confidence score for softmax

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
    app.run(host='0.0.0.0', port=port, debug=False)