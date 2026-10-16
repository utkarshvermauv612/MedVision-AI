from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "train"
    / "PNEUMONIA"
    / "person38_bacteria_196.jpeg"
)

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50


# ==========================================
# STEP 2: CHECK FILES
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Check that models/best_model_v3.keras exists."
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Check the filename and folder path."
    )


# ==========================================
# STEP 3: LOAD TRAINED MODEL
# ==========================================

print("Loading trained DenseNet121 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# STEP 4: LOAD AND PREPROCESS IMAGE
# ==========================================

print("\nLoading X-ray image...")

image = Image.open(IMAGE_PATH).convert("RGB")

# Preserve the original image for display.
original_image = image.copy()

# Resize to the model's expected input size.
image = image.resize(IMAGE_SIZE)

# Convert image to NumPy array.
image_array = np.asarray(image, dtype=np.float32)

# Apply the same preprocessing used by DenseNet121.
image_array = preprocess_input(image_array)

# Add batch dimension: (1, 224, 224, 3).
image_array = np.expand_dims(image_array, axis=0)

print("Image preprocessing completed.")
print("Input shape:", image_array.shape)


# ==========================================
# STEP 5: MAKE PREDICTION
# ==========================================

print("\nAnalyzing X-ray image...")

prediction = model.predict(image_array, verbose=0)

# This code assumes the model has one sigmoid output:
# a single pneumonia score between 0 and 1.
if prediction.shape[-1] != 1:
    raise ValueError(
        f"Expected one sigmoid output, but received "
        f"prediction shape {prediction.shape}. "
        "Check the model's output configuration."
    )

pneumonia_score = float(prediction[0][0])

# Confirm the output is a valid probability.
if not 0.0 <= pneumonia_score <= 1.0:
    raise ValueError(
        f"Model output {pneumonia_score} is outside [0, 1]. "
        "Check whether the model uses a sigmoid output."
    )

normal_score = 1.0 - pneumonia_score


# ==========================================
# STEP 6: INTERPRET PREDICTION
# ==========================================

if pneumonia_score >= THRESHOLD:
    predicted_class = "PNEUMONIA"
else:
    predicted_class = "NORMAL"

print("\n" + "=" * 45)
print("          X-RAY ANALYSIS RESULT")
print("=" * 45)

print("Image:", IMAGE_PATH.name)
print(f"Predicted class: {predicted_class}")
print(f"Model pneumonia score: {pneumonia_score * 100:.2f}%")
print(f"Model normal score:    {normal_score * 100:.2f}%")
print(f"Classification threshold: {THRESHOLD:.2f}")

print("=" * 45)

print(
    "\nIMPORTANT: This is an experimental AI prediction, "
    "not a medical diagnosis."
)


# ==========================================
# STEP 7: DISPLAY IMAGE AND RESULT
# ==========================================

plt.figure(figsize=(7, 7))

plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.title(
    f"Predicted: {predicted_class}\n"
    f"Model pneumonia score: {pneumonia_score * 100:.2f}%",
    fontsize=13
)

plt.tight_layout()
plt.show()