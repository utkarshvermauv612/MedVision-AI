from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_SIZE = (224, 224)

THRESHOLD = 0.5

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading DenseNet121 model...")
def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    # Load the uploaded image as RGB.
    original_image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Use the same image-loading approach as the
    # verified prediction pipeline.
    image = load_img(
        BytesIO(image_bytes),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(image)

    image_array = preprocess_input(
        image_array
    )

    input_array = np.expand_dims(
        image_array,
        axis=0
    )

    return original_image, input_array


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(uploaded_file):

    model = load_trained_model()

    original_image, input_array = preprocess_image(
        uploaded_file
    )

    prediction = model.predict(
        input_array,
        verbose=0
    )

    if prediction.shape != (1, 1):
        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    pneumonia_score = float(prediction[0][0])

    if not np.isfinite(pneumonia_score):
        raise ValueError("Invalid prediction score.")

    if not 0 <= pneumonia_score <= 1:
        raise ValueError(
            "Prediction score must be between 0 and 1."
        )

    normal_score = 1.0 - pneumonia_score

    predicted_class = (
        1 if pneumonia_score >= THRESHOLD else 0
    )

    predicted_label = CLASS_NAMES[predicted_class]

    return {
        "image": original_image,
        "predicted_class": predicted_label,
        "normal_score": normal_score,
        "pneumonia_score": pneumonia_score,
        "filename": uploaded_file.name,
    }