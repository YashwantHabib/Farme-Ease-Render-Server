from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import psutil  # For memory tracking
import os
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Function to print current RAM usage
def print_ram_usage(stage):
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM usage: {ram_usage:.2f} MB")


print_ram_usage("Start of script")

# Load model and define classes
MODEL_PATH = "saved_model.h5"
print_ram_usage("Before loading model")
model = tf.keras.models.load_model(MODEL_PATH)
print_ram_usage("After loading model")
CLASS_LABELS = {
    0: "Apple - Apple Scab",
    1: "Apple - Black Rot",
    2: "Apple - Cedar Apple Rust",
    3: "Apple - Healthy",
    4: "Blueberry - Healthy",
    5: "Cherry (including sour) - Healthy",
    6: "Cherry (including sour) - Powdery Mildew",
    7: "Corn (maize) - Cercospora Leaf Spot (Gray)",
    8: "Corn (maize) - Common Rust",
    9: "Corn (maize) - Healthy",
    10: "Corn (maize) - Northern Leaf Blight",
    11: "Grape - Black Rot",
    12: "Grape - Esca (Black Measles)",
    13: "Grape - Healthy",
    14: "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    15: "Orange - Huanglongbing (Citrus Greening)",
    16: "Peach - Bacterial Spot",
    17: "Peach - Healthy",
    18: "Pepper (bell) - Bacterial Spot",
    19: "Pepper (bell) - Healthy",
    20: "Potato - Early Blight",
    21: "Potato - Healthy",
    22: "Potato - Late Blight",
    23: "Raspberry - Healthy",
    24: "Soybean - Healthy",
    25: "Squash - Powdery Mildew",
    26: "Strawberry - Healthy",
    27: "Strawberry - Leaf Scorch",
    28: "Tomato - Bacterial Spot",
    29: "Tomato - Early Blight",
    30: "Tomato - Healthy",
    31: "Tomato - Late Blight",
    32: "Tomato - Leaf Mold",
    33: "Tomato - Septoria Leaf Spot",
    34: "Tomato - Spider Mites (Two-spotted Spider Mite)",
    35: "Tomato - Target Spot",
    36: "Tomato - Tomato Mosaic Virus",
    37: "Tomato - Tomato Yellow Leaf Curl Virus",
}





def preprocess_image(image_path):
    img = Image.open(image_path).resize((126, 126))  # Resize to the input size of the first Conv2D layer
    img = np.array(img).astype('float32') / 255.0    # Normalize pixel values to [0, 1]
    img = img.reshape(1, 126, 126, 3)                # Add batch dimension (1, height, width, channels)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Preprocess the image
        image_path = "temp.jpg"
        file.save(image_path)
        print_ram_usage("Before preprocessing data")
        input_data = preprocess_image(image_path)
        print_ram_usage("After preprocessing data")
        # Make the prediction
        print_ram_usage("Before making predictions")
        prediction = model.predict(input_data)
        print_ram_usage("After making predictions")
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
        predicted_label = CLASS_LABELS.get(predicted_class, "Unknown")

        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
