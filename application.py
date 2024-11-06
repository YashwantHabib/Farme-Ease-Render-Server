from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import psutil  # For memory tracking
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Load model and define classes
MODEL_PATH = "saved_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
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

# Function to print current RAM usage
def print_ram_usage(stage):
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM usage: {ram_usage:.2f} MB")

@app.route("/predict", methods=["POST"])
def predict():
    print("Received request at /predict")
    print_ram_usage("Before prediction")

    try:
        # Get preprocessed image data from the request
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Convert the image data back into a numpy array
        processed_image = np.array(data["image"], dtype=np.float32)
        if len(processed_image.shape) == 3:  # Add batch dimension if necessary
            processed_image = np.expand_dims(processed_image, axis=0)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])

        print(f"Predicted class index: {predicted_class}")
        print(f"Confidence: {confidence}")
        print(f"Number of classes: {len(CLASS_NAMES)}")

        # Check if predicted_class is within CLASS_NAMES range
        if predicted_class >= len(CLASS_NAMES):
            print("Error: Predicted class index out of range")
            return jsonify({"error": "Predicted class index out of range"}), 500

        predicted_label = CLASS_NAMES[predicted_class]
        print_ram_usage("After prediction")
        return jsonify({"label": predicted_label, "confidence": confidence}), 200

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
