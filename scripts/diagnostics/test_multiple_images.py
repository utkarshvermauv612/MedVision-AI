from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"
TEST_DIR = BASE_DIR / "chest_xray" / "test"

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50
SAMPLE_PER_CLASS = 10

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==========================================
# STEP 2: CHECK PATHS
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

for class_name in ["NORMAL", "PNEUMONIA"]:
    class_dir = TEST_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {class_dir}"
        )


# ==========================================
# STEP 3: LOAD MODEL
# ==========================================

print("Loading DenseNet121 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# STEP 4: SELECT IMAGES
# ==========================================

random.seed(42)

image_records = []

for class_name, true_label in [
    ("NORMAL", 0),
    ("PNEUMONIA", 1)
]:
    class_dir = TEST_DIR / class_name

    image_files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_files:
        raise RuntimeError(
            f"No images found in {class_dir}"
        )

    selected = random.sample(
        image_files,
        min(SAMPLE_PER_CLASS, len(image_files))
    )

    for image_path in selected:
        image_records.append((image_path, true_label))


# ==========================================
# STEP 5: PREDICT IMAGES
# ==========================================

y_true = []
y_pred = []

print(f"\nTesting {len(image_records)} images...\n")

for image_path, true_label in image_records:

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        output = model.predict(image_array, verbose=0)

        if output.shape != (1, 1):
            raise ValueError(
                f"Unexpected model output shape: {output.shape}"
            )

        pneumonia_score = float(output[0, 0])

        if not 0 <= pneumonia_score <= 1:
            raise ValueError(
                f"Output is not a probability: {pneumonia_score}"
            )

        predicted_label = int(
            pneumonia_score >= THRESHOLD
        )

        y_true.append(true_label)
        y_pred.append(predicted_label)

        expected = (
            "PNEUMONIA" if true_label == 1 else "NORMAL"
        )

        predicted = (
            "PNEUMONIA" if predicted_label == 1 else "NORMAL"
        )

        result = (
            "MATCH" if true_label == predicted_label
            else "MISMATCH"
        )

        print(f"Image: {image_path.name}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Pneumonia score: {pneumonia_score:.4f}")
        print(f"Result: {result}")
        print("-" * 45)

    except Exception as error:
        print(f"Could not process {image_path.name}: {error}")


# ==========================================
# STEP 6: EVALUATE RESULTS
# ==========================================

if not y_true:
    raise RuntimeError("No images were successfully evaluated.")

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 55)
print("             SAMPLE TEST RESULTS")
print("=" * 55)

print(f"Images evaluated: {len(y_true)}")
print(f"Correct predictions: {sum(a == b for a, b in zip(y_true, y_pred))}")
print(f"Incorrect predictions: {sum(a != b for a, b in zip(y_true, y_pred))}")
print(f"Sample accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    )
)

print("\nTesting completed.")
print(
    "Reminder: this small sample is a software check, "
    "not proof of clinical accuracy."
)