from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BEST_MODEL_PATH = (
    BASE_DIR / "models" / "best_model_v3.keras"
)

FINAL_MODEL_PATH = (
    BASE_DIR / "models" / "final_model_v3.keras"
)

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0001-0001.jpeg"
)

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50


# ============================================================
# STEP 2: VERIFY FILES
# ============================================================

print("=" * 65)
print("COMPARING SAVED MODEL FILES")
print("=" * 65)

for path in [
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    IMAGE_PATH
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nAll required files exist.")


# ============================================================
# STEP 3: PREPARE THE IMAGE
# ============================================================

img = image.load_img(
    str(IMAGE_PATH),
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(
    img_array.astype(np.float32)
)

print("\nImage:", IMAGE_PATH.name)
print("Input shape:", img_array.shape)


# ============================================================
# STEP 4: LOAD BOTH MODELS
# ============================================================

print("\nLoading best checkpoint...")

best_model = tf.keras.models.load_model(
    str(BEST_MODEL_PATH),
    compile=False
)

print("Best checkpoint loaded.")

print("\nLoading final model...")

final_model = tf.keras.models.load_model(
    str(FINAL_MODEL_PATH),
    compile=False
)

print("Final model loaded.")


# ============================================================
# STEP 5: PREDICT WITH BOTH MODELS
# ============================================================

print("\nRunning predictions...")

best_score = float(
    best_model.predict(
        img_array,
        verbose=0
    )[0][0]
)

final_score = float(
    final_model.predict(
        img_array,
        verbose=0
    )[0][0]
)


def classify(score):
    if score >= THRESHOLD:
        return "PNEUMONIA"

    return "NORMAL"


print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print("\nBEST MODEL")
print(f"Pneumonia score: {best_score:.8f}")
print(f"Predicted class: {classify(best_score)}")

print("\nFINAL MODEL")
print(f"Pneumonia score: {final_score:.8f}")
print(f"Predicted class: {classify(final_score)}")

print("\n" + "-" * 65)

print(
    "Absolute score difference:",
    f"{abs(best_score - final_score):.8f}"
)

print("\nComparison completed.")