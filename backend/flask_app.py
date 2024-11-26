from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
from joblib import load
from flask_cors import CORS
import os
import io  # Import io module

app = Flask(__name__)
CORS(app)

# Correct model path (use raw strings or forward slashes for Windows paths)
MODEL_PATH = r"C:\Users\Admin\Desktop\Projects\skincancer\backend\model\final_model.h5"

# Verify model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load the trained model (ensure the format matches the model type)
try:
    loaded_model = load(MODEL_PATH)  # If using joblib for pickle format
except Exception as e:
    raise ValueError(f"Failed to load model. Error: {e}")

# Initialize the InferenceHTTPClient with hardcoded API URL and API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Define a function to perform inference
def perform_inference(image):
    try:
        # Convert image to bytes (required for inference)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")  # Ensure proper format
        img_bytes.seek(0)

        custom_configuration = InferenceConfiguration(confidence_threshold=0.5)
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(img_bytes.getvalue(), model_id="kuchbhe/7")
        
        # Define a dictionary to map detected classes to their corresponding names
        class_names = {
            "AKIEC": "Actinic Keratosis",
            "BCC": "Basal Cell Carcinoma",
            "BKL": "Pigmented Benign Keratosis",
            "DF": "Dermatofibroma",
            "MEL": "Melanoma",
            "NV": "Nevus",
            "VASC": "Vascular Lesion"
        }
        
        # Extract detected classes and replace them with their corresponding names
        classes = [class_names.get(obj['class'], obj['class']) for obj in result.get('predictions', [])]
        return classes
    except Exception as e:
        return [f"Error during inference: {e}"]

# Define route for index page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400
        
        uploaded_image = request.files["file"]
        
        if uploaded_image.filename == "":
            return jsonify({"error": "No file selected!"}), 400
        
        try:
            # Open the uploaded image using PIL
            image = Image.open(uploaded_image)
            
            # Perform inference
            classes = perform_inference(image)
            
            # Return JSON response with the classes detected
            return jsonify({"classes": classes})
        except Exception as e:
            return jsonify({"error": f"Error processing image: {e}"}), 500
    
    # For GET requests, return a simple message
    return jsonify({"message": "Skin Cancer Detection API. POST a file to test."})

if __name__ == "__main__":
    app.run(debug=True)
