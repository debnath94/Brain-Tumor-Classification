# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:25:22 2023

@author: debna
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("your_model.h9")

def main():
    st.title("Brain Tumor Classification")
    st.text("Upload MRI scan images for tumor classification")

    # File upload
    uploaded_files = st.file_uploader("Choose MRI scan images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            image = image.resize((150, 150))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Make predictions
            prediction = model.predict(image)
            if prediction[0][0] > 0.5:
                result = "Tumor"
            else:
                result = "Normal"

            st.write("Prediction:", result)
            st.write("---")

if __name__ == "__main__":
    main()

































