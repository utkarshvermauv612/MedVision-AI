from pathlib import Path
import hashlib

import numpy as np
import tensorflow as tf

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
# LOAD IMAGE
# ============================================================

img = image.load_img(
    str(IMAGE_PATH),
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

raw_array = image.img_to_array(img).astype(np.float32)

print("=" * 65)
print("IMAGE PREPROCESSING DIAGNOSTIC")
print("=" * 65)

print("\nImage path:")
print(IMAGE_PATH.resolve())

print("\nImage dimensions:", img.size)
print("Raw array shape:", raw_array.shape)
print("Raw pixel minimum:", float(raw_array.min()))
print("Raw pixel maximum:", float(raw_array.max()))


# ============================================================
# CREATE PREPROCESSED INPUT
# ============================================================

batch_array = np.expand_dims(raw_array.copy(), axis=0)

processed_array = preprocess_input(batch_array.copy())

print("\nPreprocessed shape:", processed_array.shape)
print("Preprocessed minimum:", float(processed_array.min()))
print("Preprocessed maximum:", float(processed_array.max()))


# ============================================================
# HASH THE PREPROCESSED INPUT
# ============================================================

input_hash = hashlib.sha256(
    processed_array.tobytes()
).hexdigest()

print("\nPreprocessed input SHA-256:")
print(input_hash)


# ============================================================
# LOAD MODEL AND PREDICT
# ============================================================

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

prediction = float(
    model.predict(processed_array, verbose=0)[0][0]
)

print("\nModel path:")
print(MODEL_PATH.resolve())

print(f"\nPneumonia score: {prediction:.8f}")

print(
    "Predicted class:",
    "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
)

print("\nDiagnostic completed.")