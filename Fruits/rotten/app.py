import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('fruit_freshness_model2.h5')


# Function to predict freshness
def predict_freshness(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    freshness_score = 1 - prediction[0][0]

    # Interpret the prediction
    if freshness_score >= 0.6:
        return "Fresh", freshness_score
    else:
        return "Rotten", freshness_score


# Streamlit UI
st.title("Fruit Freshness Classifier")
st.write("Upload an image of a fruit to check its freshness.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image
    img_path = "uploaded_image.png"
    img.save(img_path)

    # Predict freshness
    result, score = predict_freshness(img_path)

    # Display the result
    st.write(f"The fruit is: **{result}**")
    st.write(f"Freshness score: {score:.2f}")

