import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.set_page_config(page_title="Face Blur App", layout="centered")

#  Custom CSS Styling 
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #3f51b5;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .uploadedImage, .processedImage {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Face Detection & Blurring 
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

#  Streamlit UI 
st.title("Face Blur Using Computer Vision")

uploaded_file = st.file_uploader("Upload an image with one or more faces", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("### Original Image")
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    image = Image.open(uploaded_file)
    result = detect_and_blur_faces(image)

    st.markdown("### Processed Image (Faces Blurred)")
    st.image(result, caption='Processed Image', use_container_width=True)

    st.success("Faces were detected and blurred successfully!")
