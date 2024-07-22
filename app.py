from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Vehicle Detection",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Vehicle DETECTION AND CLASSIFICATION USING YOLOv8")

# sidebar
st.sidebar.header("Detect vehicles")

# model options
task_type = "Detection"

model_type = None
if task_type == "Detection":
    model_type = config.DETECTION_MODEL_LIST
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = r"C:\Users\samar\OneDrive\Desktop\Animal\yolov8m.pt"
if model_type:
    #change the path of model here
    model_path = r"C:\Users\samar\OneDrive\Desktop\Animal\yolov8m.pt"
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
    print("Git")