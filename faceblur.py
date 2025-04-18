import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page configuration first
st.set_page_config(page_title="Face Blur App", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #4a4a4a;
        text-align: center;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Face Blur App")

# Function to detect and blur faces in the uploaded image
def detect_and_blur_faces(uploaded_image):
    image = np.array(uploaded_image.convert('RGB'))
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = image_cv[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(roi_color, (15, 15), 30)
        image_cv[y:y+h, x:x+w] = blurred_face
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return result_image

# Upload and display logic
uploaded_file = st.file_uploader("Upload an image with face(s)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_container_width=True)
    result = detect_and_blur_faces(image)
    st.image(result, caption='Processed Image', use_container_width=True)
    st.success("Faces detected and blurred successfully!")
