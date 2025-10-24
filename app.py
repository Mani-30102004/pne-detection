import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def load_model_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def predict(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"


st.title("Pneumonia Detection System")
st.write("Upload a Chest X-ray Image to Predict Pneumonia.")

# Resolve default model path relative to this script so Streamlit's working
# directory doesn't affect loading.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(ROOT_DIR, "pneumonia_model.h5")
model = load_model_safe(DEFAULT_MODEL)

if model is None:
    st.warning(f"Default model '{DEFAULT_MODEL}' not found or failed to load.")
    st.info("You can either train the model (run train.py) to create the file, or upload a pretrained .h5 model below.")
    uploaded_model = st.file_uploader("Upload a Keras .h5 model file", type=["h5"], key="model_upload")
    if uploaded_model:
        # Save uploaded model next to this script so subsequent runs can find it
        with open(DEFAULT_MODEL, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model = load_model_safe(DEFAULT_MODEL)

uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"], key="img_upload")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    # Save the uploaded image in the script directory as well
    uploaded_image_path = os.path.join(ROOT_DIR, "uploaded.jpg")
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if model is None:
        st.error("No model available. Please upload a .h5 model or run training (train.py) to generate 'pneumonia_model.h5'.")
    else:
        result = predict(model, uploaded_image_path)
        st.write(f"Prediction: *{result}*")