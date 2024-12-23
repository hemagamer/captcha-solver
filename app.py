import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import string
from PIL import Image
import matplotlib.pyplot as plt

# Initialize main values
symbols = string.ascii_lowercase + "0123456789"  # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

# Load the trained model
@st.cache_resource  # Cache the model for efficient reloading
def load_trained_model():
    model = tf.keras.models.load_model('cnn_model.h5')  # Ensure the model is saved locally
    return model

model = load_trained_model()

# Preprocess input image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0
    image = np.reshape(image, (50, 200, 1))
    return image

# Predict captcha
def predict_captcha(image_array):
    res = np.array(model.predict(image_array[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, num_symbols))
    captcha = ""
    for i in range(5):
        captcha += symbols[np.argmax(ans[i])]
    return captcha

# Streamlit App
st.title("Captcha Solver with CNN")
st.write("Upload a captcha image to predict its text.")

uploaded_file = st.file_uploader("Choose a captcha image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image_array = preprocess_image(image)
    predicted_text = predict_captcha(image_array)

    st.write(f"Predicted Captcha: **{predicted_text}**")
