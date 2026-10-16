from pathlib import Path

import hashlib
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0001-0001.jpeg"
)

IMAGE_SIZE = (224, 224)


# ==========================================
# STEP 2: VERIFY FILES
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Check the actual filename inside chest_xray/test/NORMAL."
    )


# ==========================================
# STEP 3: CALCULATE MODEL FILE HASH
# ==========================================

print("=" * 55)
print("MODEL CONSISTENCY CHECK")
print("=" * 55)

sha256 = hashlib.sha256()

with open(MODEL_PATH, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        sha256.update(chunk)

print("\nModel path:")
print(MODEL_PATH)

print("\nModel SHA-256:")
print(sha256.hexdigest())


# ==========================================
# STEP 4: LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ==========================================
# STEP 5: PREPROCESS ONE IMAGE
# ==========================================

print("\nProcessing image:")
print(IMAGE_PATH.name)

with Image.open(IMAGE_PATH) as image:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

print("Shape before preprocessing:", image_array.shape)

image_array = preprocess_input(image_array)

image_array = np.expand_dims(
    image_array,
    axis=0
)

print("Shape passed to model:", image_array.shape)

print("Minimum input value:", float(image_array.min()))
print("Maximum input value:", float(image_array.max()))


# ==========================================
# STEP 6: RUN PREDICTION THREE TIMES
# ==========================================

print("\nRunning prediction three times...")

predictions = []

for i in range(3):

    output = model.predict(
        image_array,
        verbose=0
    )

    print(f"Run {i + 1} raw output:", output)

    if output.shape != (1, 1):
        raise ValueError(
            f"Expected one sigmoid output, received {output.shape}"
        )

    score = float(output[0, 0])

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(
            f"Output is not a valid probability-like score: {score}"
        )

    predictions.append(score)

print("\nFinal results:")

for i, score in enumerate(predictions, start=1):
    label = "PNEUMONIA" if score >= 0.50 else "NORMAL"

    print(
        f"Run {i}: pneumonia score = {score:.6f}, "
        f"predicted class = {label}"
    )

print("\nMaximum difference between repeated runs:")
print(f"{max(predictions) - min(predictions):.10f}")

print("\nConsistency check completed.")