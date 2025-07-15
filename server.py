import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# --- Model Configuration ---
# These paths should match the files in your GitHub repository
MODEL_PATH = "prep_classifier.tflite"
LABELS_PATH = "labels.txt"

# --- Load the TFLite model and allocate tensors ---
try:
    # Use the core TensorFlow Lite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully.")
    
    # Get input and output tensor details from the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get the expected input size from the model shape
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
except Exception as e:
    print(f"🛑 Error loading model: {e}")
    interpreter = None

# --- Load labels from the labels.txt file ---
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✅ Labels loaded successfully: {labels}")
except Exception as e:
    print(f"🛑 Error loading labels: {e}")
    labels = []

# --- Define the prediction endpoint ---
# This is the URL your Flutter app will send the image to (e.g., /predict)
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model and labels were loaded correctly on startup
    if interpreter is None or not labels:
        return jsonify({'error': 'Server is not configured correctly. Check model/label paths.'}), 500

    # Check if an image file was included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read the image file from the request
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize the image to the model's expected input size
        image = image.resize((width, height))
        
        # Convert the image to a numpy array and preprocess it
        input_data = np.expand_dims(image, axis=0)
        
        # Normalize the image data to the format the model expects.
        # This normalization (from [0, 255] to [-1, 1]) is common for MobileNet models.
        # If you trained your model differently, you might need to change this.
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run the model
        interpreter.invoke()
        
        # Get the prediction result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        
        # Find the index of the highest prediction score
        top_index = np.argmax(results)
        top_label = labels[top_index]
        top_confidence = float(results[top_index])
        
        # Return the result as a simple string that the Flutter app can display
        # For a more advanced app, you might return JSON like this:
        # return jsonify({'label': top_label, 'confidence': f"{top_confidence:.1%}"})
        return f"{top_label} ({top_confidence:.1%})"

    except Exception as e:
        print(f"🛑 Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

# A simple health check endpoint to make sure the server is running
@app.route('/', methods=['GET'])
def health_check():
    return "Server is running."

# This part is for local testing. Render will use Gunicorn to run the app.
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
