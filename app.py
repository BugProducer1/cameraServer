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
class_names = ['metal', 'plastic'] # MAKE SURE THIS IS THE ORDER FROM train.py's output

try:
    print(f"Loading TFLite model from: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model loaded successfully. Expecting input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}") # Will likely be float32

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
        # (This is correct, as Keras's image_dataset_from_directory loads as RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Pre-process the image (use the RGB frame)
        img_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_batch = np.expand_dims(img_resized, axis=0)

        # !! ----- THE FIX IS HERE ----- !!
        # We must check the model's expected input type
        # If the model (from train.py) has a Rescaling(1./255) layer, 
        # it *already* does the normalization.
        # DO NOT normalize manually. Just cast to the correct type.
        
        # Check if the model expects float or int
        if input_details[0]['dtype'] == np.float32:
            # The TFLite model (with its rescaling layer) expects FLOAT pixels from 0-255
            img_batch = img_batch.astype(np.float32) 
            # *DO NOT* divide by 255.0 here. The model does it.
            # WRONG: img_batch = img_batch / 255.0 
        else:
            # Less common: if the model was quantized to expect INT8
            img_batch = img_batch.astype(np.uint8)
        
        # 3. Set tensor, invoke (run inference), and get results
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # 4. Get the result
        scores = prediction[0]
        predicted_index = np.argmax(scores)
        
        print(f"Raw Scores: {scores}") # Check these scores
        
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
    # Make sure debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)