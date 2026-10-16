from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score
)

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

VAL_DIR = BASE_DIR / "chest_xray" / "val"

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

OUTPUT_PATH = BASE_DIR / "models" / "validation_predictions_v3.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
THRESHOLD = 0.50

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# STEP 2: CHECK FILES
# ==========================================

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ==========================================
# STEP 3: LOAD VALIDATION IMAGES
# ==========================================

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=(
        tf.keras.applications.densenet.preprocess_input
    )
)

val_data = val_generator.flow_from_directory(
    directory=str(VAL_DIR),
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=CLASS_NAMES,
    shuffle=False
)

print("\nClass mapping:")
print(val_data.class_indices)

print("\nValidation images:", val_data.samples)

if val_data.samples == 0:
    raise ValueError("No validation images were found.")


# ==========================================
# STEP 4: LOAD MODEL
# ==========================================

print("\nLoading DenseNet121 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# STEP 5: PREDICT
# ==========================================

print("\nGenerating validation predictions...")

probabilities = model.predict(
    val_data,
    verbose=1
).reshape(-1)

y_true = val_data.classes

y_pred = (
    probabilities >= THRESHOLD
).astype(int)


# ==========================================
# STEP 6: METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

auc = (
    roc_auc_score(y_true, probabilities)
    if len(np.unique(y_true)) == 2
    else float("nan")
)


# ==========================================
# STEP 7: DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("DENSENET121 VALIDATION RESULTS")
print("=" * 60)

print(f"Threshold:    {THRESHOLD:.2f}")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"ROC-AUC:      {auc:.4f}")

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )
)

print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================
# STEP 8: SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "filepath": val_data.filepaths,
    "actual_label": [
        CLASS_NAMES[int(label)]
        for label in y_true
    ],
    "predicted_label": [
        CLASS_NAMES[int(label)]
        for label in y_pred
    ],
    "pneumonia_probability": probabilities
})

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nValidation predictions saved to:")
print(OUTPUT_PATH)

print("\nValidation evaluation completed.")