# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:48:07 2023

@author: debna
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tempfile
import os

model = load_model("your_model.h9")

def main():
    st.title("Brain Tumor Classification")
    st.text("Upload an MRI scan image or video for tumor classification")

    # File upload
    uploaded_file = st.file_uploader("Choose an MRI scan image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        # Check if the uploaded file is an image
        if uploaded_file.type.startswith("image"):
            # Display the uploaded image
            image = tf.image.decode_image(uploaded_file.read(), channels=3)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            image = tf.image.resize(image, (150, 150))
            image = np.expand_dims(image, axis=0) / 255.0

            # Make predictions
            prediction = model.predict(image)
            if prediction[0][0] > 0.5:
                result = "Tumor"
            else:
                result = "Normal"

            st.write("Prediction:", result)

        # Check if the uploaded file is a video
        elif uploaded_file.type.startswith("video"):
            # Save the video to a temporary file
            temp_dir = tempfile.TemporaryDirectory()
            temp_file_path = os.path.join(temp_dir.name, "temp_video.mp4")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Display the uploaded video
            video = cv2.VideoCapture(temp_file_path)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, caption="Uploaded Video Frame", use_column_width=True)

                # Preprocess the frame
                frame = cv2.resize(frame, (150, 150))
                frame = np.expand_dims(frame, axis=0) / 255.0

                # Make predictions
                prediction = model.predict(frame)
                if prediction[0][0] > 0.5:
                    result = "Tumor"
                else:
                    result = "Normal"

                st.write("Prediction:", result)

            video.release()

            # Cleanup temporary files
            temp_dir.cleanup()

if __name__ == "__main__":
    main()
