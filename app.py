import os
import cv2
import numpy as np
from flask import Flask, request, Response, render_template_string, jsonify
import base64
import io
from ultralytics import YOLO

# --- 1. INITIALIZE APP & LOAD MODEL ---

app = Flask(__name__)

# This will be the path where Render downloads the model
MODEL_PATH = "yolov8n.onnx"
model = None

# List of object names YOLOv8n can detect
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Try loading the model when the application starts
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading YOLO model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Check build command.")
except Exception as e:
    print(f"ERROR: Failed to load YOLO model: {e}")
    model = None # Ensure model is None if loading failed

# --- 2. YOLO DETECTION FUNCTION ---

def detect_objects_in_image(frame):
    if model is None:
        print("[Error] YOLO model not loaded.")
        return [], frame, "Error: Model not loaded"

    detections = []
    plain_text_results = []
    
    try:
        # Run YOLO detection
        results = model(frame)

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = class_names[cls_id]

                if confidence > 0.5: # Confidence threshold
                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "box": [x1, y1, x2, y2]
                    })
                    plain_text_results.append(f"{label} ({confidence*100:.0f}%)")

                    # Draw box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence*100:.0f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                                
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return [], frame, "Error: Inference failed"

    result_string = ", ".join(plain_text_results) if plain_text_results else "No objects detected."
    return detections, frame, result_string

# --- 3. FLASK WEB ROUTES ---

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection</title>
     <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 40px; background: #f7f7f7; }
        h1 { color: #333; }
        form { margin: 20px auto; padding: 20px; background: #fff; border-radius: 10px; max-width: 500px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        input[type="file"] { border: 1px solid #ccc; padding: 10px; border-radius: 5px; width: calc(100% - 22px); margin-bottom: 10px;}
        input[type="submit"] { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.2s; }
        input[type="submit"]:hover { background: #0056b3; }
        img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 5px; border: 1px solid #ddd; }
        .result { font-weight: bold; color: #333; margin-top: 15px; font-size: 18px; padding: 10px; background-color: #e9ecef; border-radius: 5px;}
        .preview-container { margin-top: 20px; max-width: 800px; margin-left: auto; margin-right: auto;}
         .error { color: red; font-weight: bold; margin-top: 15px; }
    </style>
</head>
<body>
    <h1>YOLO Object Detector</h1>
    <form action="/preview" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br>
        <input type="submit" value="Upload & Detect">
    </form>
    {% if error %}
        <div class="error">Error: {{ error }}</div>
    {% elif image_url %}
        <div class="preview-container">
            <img src="{{ image_url }}" alt="Uploaded Image Preview with Detections">
            <div class="result">Detections: {{ result }}</div>
        </div>
    {% elif result %}
         <div class="result">{{ result }}</div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/preview', methods=['POST'])
def preview():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model failed to load. Check server logs.")
        
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template_string(HTML_TEMPLATE, result="No file selected")

        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
             return render_template_string(HTML_TEMPLATE, error="Could not decode image.")

        detections, frame_with_boxes, result_string = detect_objects_in_image(frame)

        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"

        return render_template_string(HTML_TEMPLATE, result=result_string, image_url=image_url)

    except Exception as e:
        print(f"Error in /preview: {e}")
        return render_template_string(HTML_TEMPLATE, error=f"Server error during processing: {e}")

@app.route('/upload', methods=['POST'])
def upload_image():
    if model is None:
         return jsonify({"error": "Model not loaded"}), 500
         
    try:
        image_data = request.data
        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        detections, _, _ = detect_objects_in_image(frame)

        return jsonify(detections)

    except Exception as e:
        print(f"Error in /upload: {e}")
        return jsonify({"error": "Server error during processing"}), 500

# --- 4. RUN THE APP ---

if __name__ == '__main__':
    # This block runs when you run `python app.py` locally
    port = int(os.environ.get('PORT', 10000))
    # Note: Model loading is handled at the top level now
    app.run(host='0.0.0.0', port=port, debug=True) # Turn debug=True for local testing

# Gunicorn on Render will run this file, loading the model at the top.