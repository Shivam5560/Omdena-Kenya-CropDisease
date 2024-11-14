import streamlit as st
import json
import subprocess
import os
from PIL import Image
import os
import shutil

# Define the directory path
directory_path = '/runs/predict-cls'

# List all items in the directory
for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    # Check if the item is a folder
    if os.path.isdir(item_path):
        # Remove the folder and its contents
        shutil.rmtree(item_path)
    else:
        
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
    start_idx = output_text.find("224x224") + 8
    val = output_text[start_idx:].split(',')[:5]
    d = {}
    for x in val:
        a = x.strip().split(' ')
        d[a[0]]= float(a[1])
    with st.sidebar:
        st.write("Results")
        st.json(d)


    # Clean up temporary image
    os.remove(img_path)
