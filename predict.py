import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

# predict.py is directly inside the project root.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

THRESHOLD = 0.5


# ============================================================
# STEP 2: LOAD TRAINED MODEL
# ============================================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nTrained model not found!\n"
            f"Expected location: {MODEL_PATH}\n"
            f"Check that best_model_v3.keras exists "
            f"inside your models folder."
        )

    print("=" * 60)
    print("CHEST X-RAY PREDICTION SYSTEM")
    print("=" * 60)

    print(f"\nProject directory: {BASE_DIR}")
    print(f"Model path: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("\nTrained DenseNet121 model loaded successfully!")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    return model


# ============================================================
# STEP 3: PREPROCESS INPUT IMAGE
# ============================================================

def preprocess_image(image_path):

    image_path = Path(image_path)

    # If a relative path is supplied, first check the current
    # working directory, then the project directory.
    if not image_path.is_absolute():

        current_directory_path = image_path.resolve()

        project_directory_path = BASE_DIR / image_path

        if current_directory_path.exists():
            image_path = current_directory_path

        elif project_directory_path.exists():
            image_path = project_directory_path

        else:
            raise FileNotFoundError(
                f"\nImage not found!\n"
                f"Checked: {current_directory_path}\n"
                f"Checked: {project_directory_path}"
            )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load the image using the same image-loading approach
    # used in the model verification pipeline.
    original_image = load_img(
        str(image_path),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(original_image)

    # DenseNet121 preprocessing.
    # Do NOT divide by 255.0 separately.
    image_array = preprocess_input(image_array)

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    input_image = np.expand_dims(
        image_array,
        axis=0
    )

    print("\nImage loaded and preprocessed successfully.")
    print("Image path:", image_path)
    print("Input tensor shape:", input_image.shape)

    return image_path, original_image, input_image


# ============================================================
# STEP 4: PREDICT IMAGE CLASS
# ============================================================

def predict_image(image_path, model):

    image_path, original_image, input_image = preprocess_image(
        image_path
    )

    # Predict the pneumonia score.
    prediction = model.predict(
        input_image,
        verbose=0
    )

    # This script assumes the model has one sigmoid output:
    # NORMAL = 0, PNEUMONIA = 1.
    if prediction.shape != (1, 1):
        raise ValueError(
            "Unexpected model output shape: "
            f"{prediction.shape}. "
            "This script expects a single sigmoid output."
        )

    pneumonia_score = float(prediction[0][0])

    if not np.isfinite(pneumonia_score):
        raise ValueError(
            "The model returned an invalid prediction score."
        )

    if not 0.0 <= pneumonia_score <= 1.0:
        raise ValueError(
            "The model output is outside the expected "
            "sigmoid score range [0, 1]."
        )

    normal_score = 1.0 - pneumonia_score

    predicted_class = (
        1 if pneumonia_score >= THRESHOLD else 0
    )

    predicted_label = CLASS_NAMES[predicted_class]

    # ========================================================
    # STEP 5: PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(f"Image: {image_path.name}")

    print(f"Predicted class: {predicted_label}")

    print(
        f"Model score for NORMAL:    {normal_score:.4f} "
        f"({normal_score * 100:.2f}%)"
    )

    print(
        f"Model score for PNEUMONIA: {pneumonia_score:.4f} "
        f"({pneumonia_score * 100:.2f}%)"
    )

    print(f"Classification threshold: {THRESHOLD:.2f}")

    print("=" * 60)

    print(
        "\nIMPORTANT: These are model scores, not necessarily "
        "calibrated probabilities. This experimental model "
        "is not a substitute for professional medical diagnosis."
    )

    # ========================================================
    # STEP 6: DISPLAY IMAGE AND PREDICTION
    # ========================================================

    plt.figure(figsize=(8, 7))

    plt.imshow(original_image)

    plt.title(
        f"Predicted Class: {predicted_label}\n"
        f"Pneumonia Model Score: {pneumonia_score:.3f}",
        fontsize=14
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_label,
        "normal_score": normal_score,
        "pneumonia_score": pneumonia_score,
        "threshold": THRESHOLD
    }


# ============================================================
# STEP 7: RUN PREDICTION FROM TERMINAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict NORMAL or PNEUMONIA from "
            "a chest X-ray using DenseNet121."
        )
    )

    parser.add_argument(
        "image_path",
        help="Path to the chest X-ray image."
    )

    args = parser.parse_args()

    model = load_trained_model()

    predict_image(
        args.image_path,
        model
    )


if __name__ == "__main__":
    main()