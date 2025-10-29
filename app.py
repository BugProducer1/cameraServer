import os
import cv2
import numpy as np
import tensorflow as tf
tflite = tf.lite
from flask import Flask, request, Response, render_template_string
import base64

app = Flask(__name__)

MODEL_PATH = "model.tflite"
interpreter = None
input_details = None
output_details = None
IMG_HEIGHT = 96
IMG_WIDTH = 96
class_names = ['plastic', 'metal']

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Model load error: {e}")

def run_ai_on_image(image_bytes):
    if interpreter is None:
        return "Error: Model not loaded", None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return "Error: Bad image data", None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_batch = np.expand_dims(img_resized, axis=0)
        if input_details[0]['dtype'] == np.float32:
            img_batch = img_batch.astype(np.float32) / 255.0
        else:
            img_batch = img_batch.astype(np.float32) / 255.0
            interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        scores = prediction[0]
        idx = np.argmax(scores)
        label = class_names[idx]
        conf = 100 * float(scores[idx])
        return f"{label} ({conf:.2f}%)", frame_rgb
    except Exception as e:
        return "Error: Inference failed", None

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>AI Image Preview & Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 40px; background: #f7f7f7; }
        form { margin: 20px auto; padding: 20px; background: #fff; border-radius: 10px; width: 400px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        img { max-width: 300px; margin-top: 20px; border-radius: 5px; }
        .result { font-weight: bold; color: #0066cc; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>AI Image Classifier</h1>
    <form action="/preview" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br><br>
        <input type="submit" value="Upload & Predict">
    </form>
    {% if image_url %}
        <div>
            <img src="{{ image_url }}" alt="Uploaded Image Preview">
            <div class="result">{{ result }}</div>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/preview', methods=['POST'])
def preview():
    try:
        file = request.files['file']
        if not file:
            return render_template_string(HTML_TEMPLATE, result="No file selected")
        image_bytes = file.read()
        result, frame_rgb = run_ai_on_image(image_bytes)
        if frame_rgb is not None:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_url = f"data:image/jpeg,jpg;base64,{image_base64}"
        else:
            image_url = None
        return render_template_string(HTML_TEMPLATE, result=result, image_url=image_url)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result="Server error")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image_data = request.data
        if not image_data:
            return Response(response="Error: No image data received", status=400, mimetype='text/plain')
        result, _ = run_ai_on_image(image_data)
        return Response(response=result, status=200, mimetype='text/plain')
    except Exception as e:
        return Response(response="Server Error", status=500, mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)