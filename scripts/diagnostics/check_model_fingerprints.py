import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "BEST": BASE_DIR / "models" / "best_model_v3.keras",
    "FINAL": BASE_DIR / "models" / "final_model_v3.keras",
}

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0001-0001.jpeg"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# SHA-256
# ============================================================

def sha256_file(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# VERIFY FILES
# ============================================================

print("=" * 70)
print("MODEL FINGERPRINT AND PREDICTION CHECK")
print("=" * 70)

for name, path in MODEL_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(path)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(IMAGE_PATH)


# ============================================================
# PRINT PATHS AND HASHES
# ============================================================

print("\nPROJECT DIRECTORY:")
print(BASE_DIR)

print("\nIMAGE PATH:")
print(IMAGE_PATH.resolve())

print("\nMODEL FINGERPRINTS:")

for name, path in MODEL_PATHS.items():
    print(f"\n{name} MODEL")
    print("Path:", path.resolve())
    print("Size in bytes:", path.stat().st_size)
    print("SHA-256:", sha256_file(path))


# ============================================================
# PREPARE IMAGE ONCE
# ============================================================

img = image.load_img(
    str(IMAGE_PATH),
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

array = image.img_to_array(img).astype(np.float32)

array = np.expand_dims(array, axis=0)

array = preprocess_input(array)

print("\nPREPROCESSING:")
print("Shape:", array.shape)
print("Minimum:", float(array.min()))
print("Maximum:", float(array.max()))


# ============================================================
# PREDICT WITH EACH MODEL
# ============================================================

for name, path in MODEL_PATHS.items():

    print(f"\n{'=' * 50}")
    print(f"{name} MODEL PREDICTION")
    print("=" * 50)

    model = tf.keras.models.load_model(
        str(path),
        compile=False
    )

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    score = float(
        model.predict(array, verbose=0).reshape(-1)[0]
    )

    print(f"Pneumonia score: {score:.8f}")
    print(
        "Predicted class:",
        "PNEUMONIA" if score >= 0.5 else "NORMAL"
    )

    del model


print("\nCHECK COMPLETED.")