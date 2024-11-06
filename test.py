import requests
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import json

# URL of the Flask server
url = "http://127.0.0.1:5000/predict"

def load_and_preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    img_size = (128, 128)
    
    # Resize and preprocess the image
    image = cv2.resize(image, img_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    
    # Convert the image to a list format for JSON serialization
    image_list = image.tolist()
    return image_list

# Image path
image_path = r"C:\Users\91984\Desktop\ALL\work\college\dataset\test\test\CornCommonRust1.JPG"
img_data = load_and_preprocess_image(image_path)

# Prepare the JSON payload
payload = {
    "image": img_data
}

# Send the POST request with JSON data
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    prediction = response.json()
    print("Predicted Label:", prediction["label"])
    print("Confidence:", prediction["confidence"])
else:
    print("Error:", response.json())
