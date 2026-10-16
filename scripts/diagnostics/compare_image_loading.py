from pathlib import Path

import hashlib
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0001-0001.jpeg"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model_v3.keras"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# VERIFY FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(IMAGE_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


# ============================================================
# METHOD A: KERAS IMAGE LOADER
# ============================================================

keras_img = image.load_img(
    str(IMAGE_PATH),
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

keras_array = image.img_to_array(
    keras_img
).astype(np.float32)


# ============================================================
# METHOD B: PIL IMAGE LOADER
# ============================================================

with Image.open(IMAGE_PATH) as pil_img:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMAGE_SIZE)

    pil_array = np.asarray(
        pil_img,
        dtype=np.float32
    )


# ============================================================
# COMPARE RAW PIXELS
# ============================================================

print("=" * 65)
print("IMAGE LOADING COMPARISON")
print("=" * 65)

print("\nImage:", IMAGE_PATH.name)

print("\nKeras raw pixels:")
print("Shape:", keras_array.shape)
print("Minimum:", float(keras_array.min()))
print("Maximum:", float(keras_array.max()))

print("\nPillow raw pixels:")
print("Shape:", pil_array.shape)
print("Minimum:", float(pil_array.min()))
print("Maximum:", float(pil_array.max()))

raw_difference = np.abs(
    keras_array - pil_array
)

print("\nRaw pixel comparison:")
print("Identical arrays:", np.array_equal(keras_array, pil_array))
print("Maximum absolute difference:", float(raw_difference.max()))
print("Mean absolute difference:", float(raw_difference.mean()))


# ============================================================
# APPLY IDENTICAL DENSENET PREPROCESSING
# ============================================================

keras_input = preprocess_input(
    keras_array[np.newaxis, ...].copy()
)

pil_input = preprocess_input(
    pil_array[np.newaxis, ...].copy()
)


# ============================================================
# COMPARE PREPROCESSED INPUTS
# ============================================================

print("\nPreprocessed inputs:")

print(
    "Keras input range:",
    float(keras_input.min()),
    "to",
    float(keras_input.max())
)

print(
    "Pillow input range:",
    float(pil_input.min()),
    "to",
    float(pil_input.max())
)

input_difference = np.abs(
    keras_input - pil_input
)

print(
    "Identical preprocessed arrays:",
    np.array_equal(keras_input, pil_input)
)

print(
    "Maximum preprocessed difference:",
    float(input_difference.max())
)

print(
    "Mean preprocessed difference:",
    float(input_difference.mean())
)


# ============================================================
# LOAD THE SAME MODEL ONCE
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)


# ============================================================
# PREDICT WITH BOTH INPUTS
# ============================================================

keras_score = float(
    model.predict(
        keras_input,
        verbose=0
    )[0][0]
)

pil_score = float(
    model.predict(
        pil_input,
        verbose=0
    )[0][0]
)

print("\n" + "=" * 65)
print("PREDICTION COMPARISON")
print("=" * 65)

print(f"Keras loader score:  {keras_score:.8f}")
print(f"Pillow loader score: {pil_score:.8f}")

print(
    "Absolute score difference:",
    f"{abs(keras_score - pil_score):.8f}"
)

print(
    "Keras predicted class:",
    "PNEUMONIA" if keras_score >= 0.5 else "NORMAL"
)

print(
    "Pillow predicted class:",
    "PNEUMONIA" if pil_score >= 0.5 else "NORMAL"
)

print("\nComparison completed.")