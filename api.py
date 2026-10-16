from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import img_to_array

from src.image_validator import validate_uploaded_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_SIZE = (224, 224)

THRESHOLD = 0.5

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

MAX_IMAGE_SIZE_MB = 10


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_IMAGE_SIZE_MB * 1024 * 1024
)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:8501",
                "http://127.0.0.1:8501",
            ]
        }
    },
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("MedVisionAI")


# ============================================================
# 3. MODEL LOADING
# ============================================================

model = None


def load_trained_model():
    """Load the existing trained DenseNet121 model."""

    global model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    logger.info("Loading DenseNet121 model...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    logger.info("DenseNet121 model loaded successfully.")


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_validated_image(image):
    """
    Preprocess an already validated PIL image.

    Keep DenseNet121 preprocessing consistent with the
    existing model pipeline.
    """

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        resample=2,  # PIL bilinear resampling
    )

    image_array = img_to_array(image)

    image_array = preprocess_input(image_array)

    input_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return input_array


# ============================================================
# 5. PREDICTION
# ============================================================

def predict_uploaded_image(image):
    """Predict NORMAL versus PNEUMONIA."""

    if model is None:
        raise RuntimeError(
            "The AI model has not been loaded."
        )

    input_array = preprocess_validated_image(image)

    predictions = model.predict(
        input_array,
        verbose=0,
    )

    predictions = np.asarray(predictions)

    if predictions.shape != (1, 1):
        raise RuntimeError(
            "Unexpected model output shape: "
            f"{predictions.shape}. "
            "Expected a single sigmoid output."
        )

    pneumonia_score = float(predictions[0, 0])

    if not np.isfinite(pneumonia_score):
        raise RuntimeError(
            "The model returned an invalid prediction."
        )

    if not 0.0 <= pneumonia_score <= 1.0:
        raise RuntimeError(
            "The model output is outside the probability range."
        )

    normal_score = 1.0 - pneumonia_score

    predicted_class = (
        "PNEUMONIA"
        if pneumonia_score >= THRESHOLD
        else "NORMAL"
    )

    confidence = max(
        normal_score,
        pneumonia_score,
    )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "normal_score": round(normal_score, 6),
        "pneumonia_score": round(pneumonia_score, 6),
    }


# ============================================================
# 6. HOME ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "MedVision AI",
        "message": "Flask Medical Image Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/api/predict",
        },
    }), 200


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    status_code = 200 if model is not None else 503

    return jsonify({
        "status": (
            "healthy"
            if model is not None
            else "unavailable"
        ),
        "model_loaded": model is not None,
    }), status_code


# ============================================================
# 8. PREDICTION ENDPOINT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "The trained model is unavailable.",
        }), 503

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": (
                "No image provided. "
                "Send the image using the 'image' field."
            ),
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "No file was selected.",
        }), 400

    filename = Path(uploaded_file.filename).name

    try:
        file_bytes = uploaded_file.read()

        if not file_bytes:
            return jsonify({
                "success": False,
                "error": "The uploaded file is empty.",
            }), 400

        # ----------------------------------------------------
        # STEP A: Validate the uploaded image
        # ----------------------------------------------------

        validation = validate_uploaded_image(file_bytes)

        if not validation["valid"]:

            logger.info(
                "Image rejected by quality validation: %s",
                filename,
            )

            return jsonify({
                "success": False,
                "error": (
                    "Image validation failed. "
                    "Please upload a clearer image."
                ),
                "validation_errors": validation["errors"],
                "quality": validation["quality"],
            }), 400

        # ----------------------------------------------------
        # STEP B: Run the existing model
        # ----------------------------------------------------

        image = validation["image"]

        result = predict_uploaded_image(image)

        # ----------------------------------------------------
        # STEP C: Return the prediction
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "application": "MedVision AI",
            "filename": filename,
            "prediction": result,
            "image_quality": validation["quality"],
            "notice": (
                "Educational research prototype only. "
                "This result is not a medical diagnosis."
            ),
        }), 200

    except Exception:
        logger.exception("Prediction failed.")

        return jsonify({
            "success": False,
            "error": (
                "Prediction failed. "
                "Check the Flask server logs."
            ),
        }), 500


# ============================================================
# 9. ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def handle_large_file(error):
    return jsonify({
        "success": False,
        "error": (
            f"File too large. Maximum size is "
            f"{MAX_IMAGE_SIZE_MB} MB."
        ),
    }), 413


@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({
        "success": False,
        "error": "API endpoint not found.",
    }), 404


# ============================================================
# 10. START SERVER
# ============================================================

if __name__ == "__main__":

    try:
        load_trained_model()

    except Exception:
        logger.exception("Could not load the trained model.")
        raise SystemExit(1)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )