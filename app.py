from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})  # Enable CORS for cross-origin requests

# Load your model (ensure the model is in the same directory or provide the correct path)
MODEL_PATH = "saved_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define your class names
CLASS_NAMES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
               "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
               "Corn_(maize)___Cercospora_leaf_spot Gray", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
               "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
               "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
               "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
               "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
               "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
               "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
               "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
               "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# Image preprocessing function
def load_and_preprocess_image(image):
    img_size = (128, 128)
    image = cv2.resize(image, img_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        # Read the image and preprocess it
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        processed_image = load_and_preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])

        # Get the class label and confidence
        predicted_label = CLASS_NAMES[predicted_class]
        return jsonify({"label": predicted_label, "confidence": confidence}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
