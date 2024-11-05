import requests

# URL of the local Flask server
url = "http://127.0.0.1:5000/predict"

# Open the image file in binary mode
with open(r"C:\Users\91984\Desktop\ALL\work\college\dataset\test\test\CornCommonRust1.JPG", "rb") as image_file:
    # Prepare the files dictionary for the POST request
    files = {"file": image_file}
    
    # Send the POST request to the server
    response = requests.post(url, files=files)
    
    # Check the response
    if response.status_code == 200:
        prediction = response.json()
        print("Predicted Label:", prediction["label"])
        print("Confidence:", prediction["confidence"])
    else:
        print("Error:", response.json())
