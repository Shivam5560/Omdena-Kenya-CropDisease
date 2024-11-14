import streamlit as st
import json
import subprocess
import os
from PIL import Image
import os
import shutil

# Define the directory path relative to the script's location
base_directory = os.path.dirname(os.path.abspath(__file__))
print(base_directory)
directory_path = os.path.join(base_directory, 'runs/predict-cls')

# List all items in the directory
for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    # Check if the item is a folder
    if os.path.isdir(item_path):
        # Remove the folder and its contents
        shutil.rmtree(item_path)
    else:
        os.remove(item_path)

print("All items in the directory have been removed.")
        
# Set the path to your weights and model file
WEIGHTS_PATH = "/classification_project/yolo_cassava_classification2/weights/best.pt"

st.title("Cassava Disease Classification")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image for model input
    img_path = "temp_image.jpg"
    img = Image.open(uploaded_file)
    st.image(img)
    img.save(img_path)

    # Run YOLOv5 classification command
    command = [
        "python", "classify/predict.py",
        "--weights", WEIGHTS_PATH,
        "--source", img_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Parse the output to get class labels and confidence scores
    output_text = result.stderr
    st.write(output_text)
    # Clean up temporary image
    os.remove(img_path)
