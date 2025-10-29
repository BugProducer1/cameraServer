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

# List of object names YOLOv8n can detect (COCO dataset)
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
        # Explicitly mention task if needed, though usually inferred for ONNX
        model = YOLO(MODEL_PATH, task='detect') 
        print("YOLO model loaded successfully.")
    else:
        # This error is critical if the build command didn't work
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. Check Render build command and logs.")
        model = None # Ensure model stays None
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load YOLO model: {e}")
    model = None # Ensure model stays None

# --- 2. YOLO DETECTION FUNCTION ---

def detect_objects_in_image(frame):
    """
    Performs object detection on an image frame using the loaded YOLO model.
    Returns:
        tuple: (list of detections, frame with boxes drawn, string summary of results)
    """
    # Check if the model loaded successfully during startup
    if model is None:
        print("[Error] YOLO model is not loaded. Cannot perform detection.")
        return [], frame, "Error: Model not loaded or failed to load"

    detections = []  # List to store detected object info (dict)
    plain_text_results = [] # List to store simple text results for display

    try:
        # Run YOLO inference
        results = model(frame) # The ultralytics library handles pre/post processing

        # Process results list
        for r in results:
            boxes = r.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                # Extract box coordinates (convert tensor to numpy, then int)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Extract confidence score (convert tensor to float)
                confidence = float(box.conf[0].cpu().numpy())
                
                # Extract class ID (convert tensor to int)
                cls_id = int(box.cls[0].cpu().numpy())

                # --- START DEBUGGING ---
                print(f"DEBUG: Detected Class ID: {cls_id}")
                # Look up the class name using the ID
                if 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]
                    print(f"DEBUG: Mapped Label: {label}")
                else:
                    # This should not happen with a standard model but is a good safeguard
                    label = "Unknown ID"
                    print(f"DEBUG: Class ID {cls_id} is out of bounds for class_names list!")
                # --- END DEBUGGING ---

                # Filter detections based on confidence threshold
                if confidence > 0.5:
                    # Append detection details to lists
                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2), # Round for cleaner JSON
                        "box": [x1, y1, x2, y2] # Store coordinates
                    })
                    # Format for simple text display
                    text_result = f"{label} ({confidence*100:.0f}%)"
                    plain_text_results.append(text_result)

                    # --- Draw bounding box and label on the image ---
                    # Box rectangle (green color, thickness 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Label text (place above the box)
                    cv2.putText(frame, text_result,
                                (x1, y1 - 10), # Position offset slightly above the top-left corner
                                cv2.FONT_HERSHEY_SIMPLEX, # Font type
                                0.6, # Font scale
                                (0, 255, 0), # Font color (green)
                                2) # Font thickness

    except Exception as e:
        # Log any errors during the inference process
        print(f"Error during YOLO inference: {e}")
        return [], frame, "Error: Inference failed. Check server logs."

    # Create a summary string for the HTML page
    result_string = ", ".join(plain_text_results) if plain_text_results else "No objects detected."
    # Return all processed data
    return detections, frame, result_string

# --- 3. FLASK WEB ROUTES ---

# HTML template for the browser interface
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection</title>
     <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 40px; background: #f7f7f7; margin: 0;}
        .container { max-width: 900px; margin: 20px auto; padding: 20px; background: #fff; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-top: 0; }
        form { margin-bottom: 20px; padding: 20px; border: 1px dashed #ccc; border-radius: 5px; background: #fdfdfd; }
        input[type="file"] { border: 1px solid #ccc; padding: 10px; border-radius: 5px; width: calc(100% - 24px); margin-bottom: 10px; box-sizing: border-box; }
        input[type="submit"] { background: #007bff; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.2s; }
        input[type="submit"]:hover { background: #0056b3; }
        img { max-width: 100%; height: auto; margin-top: 20px; border-radius: 5px; border: 1px solid #ddd; display: block; margin-left: auto; margin-right: auto;}
        .result { font-weight: bold; color: #333; margin-top: 15px; font-size: 18px; padding: 10px; background-color: #e9ecef; border-radius: 5px; word-wrap: break-word; }
        .preview-container { margin-top: 20px; }
        .error { color: #dc3545; font-weight: bold; margin-top: 15px; background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
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
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page for browser testing."""
    # Renders the HTML template when someone visits the root URL
    return render_template_string(HTML_TEMPLATE)

@app.route('/preview', methods=['POST'])
def preview():
    """Handles image uploads from the browser, runs detection, and shows results."""
    # Check if the model is loaded before processing
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model failed to load during server startup. Check server logs.")
        
    try:
        # Get the uploaded file from the form
        file = request.files.get('file')
        # Basic validation: check if a file was actually selected
        if not file or file.filename == '':
            return render_template_string(HTML_TEMPLATE, result="No file selected. Please choose an image.")

        # Read the image file bytes
        image_bytes = file.read()
        # Convert bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the NumPy array into an OpenCV image (in BGR format)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Check if decoding was successful
        if frame is None:
             return render_template_string(HTML_TEMPLATE, error="Could not decode image. Please upload a valid image file (JPG, PNG, etc.).")

        # Perform object detection
        detections, frame_with_boxes, result_string = detect_objects_in_image(frame)

        # --- Prepare image for display in HTML ---
        # Encode the image (with boxes drawn) back into JPG bytes
        retval, buffer = cv2.imencode('.jpg', frame_with_boxes)
        if not retval:
            return render_template_string(HTML_TEMPLATE, error="Could not encode result image.")
        # Convert the JPG bytes to a Base64 string
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        # Create a Data URL for embedding in the HTML img tag
        image_url = f"data:image/jpeg;base64,{image_base64}"

        # Render the HTML template again, passing the results
        return render_template_string(HTML_TEMPLATE, result=result_string, image_url=image_url)

    except Exception as e:
        # Log unexpected errors and inform the user
        print(f"Error in /preview endpoint: {e}")
        return render_template_string(HTML_TEMPLATE, error=f"An unexpected server error occurred: {e}")

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Handles image uploads from ESP32-CAM or other programs.
    Returns detection results as JSON.
    """
     # Check if the model is loaded before processing
    if model is None:
         # Return a server error if the model isn't ready
         return jsonify({"error": "Model not loaded on server."}), 500
         
    try:
        # Get the raw image data from the request body
        image_data = request.data
        # Basic validation
        if not image_data:
            return jsonify({"error": "No image data received in request body."}), 400

        # Convert bytes to NumPy array and decode into OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Check if decoding worked
        if frame is None:
            return jsonify({"error": "Could not decode image data."}), 400

        # Perform object detection (we only need the list of detections, not the image)
        detections, _, _ = detect_objects_in_image(frame)

        # Return the list of detected objects as a JSON response
        return jsonify(detections)

    except Exception as e:
        # Log unexpected errors and return a generic server error
        print(f"Error in /upload endpoint: {e}")
        return jsonify({"error": "Server error during processing."}), 500

# --- 4. RUN THE APP ---

# This block is used when running the script directly (e.g., `python app.py`)
if __name__ == '__main__':
    print("Starting Flask development server...")
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your network
    # debug=True provides helpful error messages during development (DISABLE for production/Render)
    app.run(host='0.0.0.0', port=port, debug=False) 

# When deployed on Render using Gunicorn, Gunicorn runs this file as a module.
# The code outside the `if __name__ == '__main__':` block, including model loading,
# will execute when Gunicorn starts each worker process.