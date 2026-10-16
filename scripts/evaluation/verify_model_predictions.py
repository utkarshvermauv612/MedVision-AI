import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "chest_xray" / "test"

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

OUTPUT_DIR = BASE_DIR / "models"

PREDICTIONS_CSV = OUTPUT_DIR / "verification_predictions_v3.csv"

METRICS_JSON = OUTPUT_DIR / "verification_metrics_v3.json"

CONFUSION_MATRIX_PNG = OUTPUT_DIR / "verification_confusion_matrix_v3.png"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

THRESHOLD = 0.5

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

CLASS_TO_LABEL = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


# ============================================================
# STEP 2: CHECK REQUIRED FILES
# ============================================================

print("=" * 65)
print("MEDICAL IMAGE MODEL VERIFICATION")
print("=" * 65)

print(f"\nProject directory: {BASE_DIR}")
print(f"Test directory:    {TEST_DIR}")
print(f"Model path:        {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 3: COLLECT TEST IMAGES
# ============================================================

print("\nCollecting test images...")

records = []

for class_name in CLASS_NAMES:

    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class directory not found:\n{class_dir}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    for image_path in image_paths:

        records.append(
            {
                "filepath": str(image_path.resolve()),
                "filename": image_path.name,
                "actual_label": CLASS_TO_LABEL[class_name],
                "actual_class": class_name,
            }
        )

if not records:
    raise ValueError(
        f"No supported images found in:\n{TEST_DIR}"
    )

# Preserve deterministic ordering.
records = sorted(
    records,
    key=lambda record: (
        record["actual_label"],
        record["filename"].lower(),
    ),
)

df = pd.DataFrame(records)

print(f"\nTotal test images: {len(df)}")

print("\nActual class distribution:")

print(
    df["actual_class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)


# ============================================================
# STEP 4: LOAD TRAINED MODEL
# ============================================================

print("\nLoading DenseNet121 model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

print("Model loaded successfully.")

print(f"Input shape:  {model.input_shape}")
print(f"Output shape: {model.output_shape}")

if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError(
        f"Unexpected model input shape: {model.input_shape}"
    )


# ============================================================
# STEP 5: LOAD AND PREPROCESS IMAGES
# ============================================================

def load_and_preprocess_image(image_path):
    """
    Load an image using Keras and apply DenseNet preprocessing.

    The image-loading method and preprocessing must match
    the model's training pipeline.
    """

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image_array = image.img_to_array(img)

    # Apply DenseNet preprocessing exactly once.
    image_array = preprocess_input(image_array)

    return image_array


# ============================================================
# STEP 6: GENERATE PREDICTIONS IN BATCHES
# ============================================================

print("\nGenerating predictions...")

all_probabilities = []

total_images = len(df)

for start in range(0, total_images, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_images)

    batch_records = records[start:end]

    batch_images = []

    for record in batch_records:

        try:
            processed_image = load_and_preprocess_image(
                record["filepath"]
            )

            batch_images.append(processed_image)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image:\n"
                f"{record['filepath']}\n"
                f"Error: {exc}"
            ) from exc

    batch_array = np.asarray(
        batch_images,
        dtype=np.float32,
    )

    # Model output is expected to be one sigmoid probability
    # per image, representing PNEUMONIA.
    batch_predictions = model.predict(
        batch_array,
        verbose=0,
    )

    if (
        batch_predictions.ndim != 2
        or batch_predictions.shape[1] != 1
    ):
        raise ValueError(
            "Expected model output shape (batch_size, 1), "
            f"but received {batch_predictions.shape}"
        )

    batch_probabilities = batch_predictions[:, 0]

    all_probabilities.extend(
        batch_probabilities.astype(float).tolist()
    )

    print(
        f"Processed {end}/{total_images} images",
        end="\r",
    )

print("\n\nPrediction generation completed.")


# ============================================================
# STEP 7: VALIDATE PREDICTIONS
# ============================================================

y_true = df["actual_label"].to_numpy(dtype=int)

y_prob = np.asarray(
    all_probabilities,
    dtype=np.float64,
)

if len(y_prob) != len(y_true):
    raise ValueError(
        "Prediction count does not match the number of images."
    )

if not np.all(np.isfinite(y_prob)):
    raise ValueError(
        "Predictions contain NaN or infinite values."
    )

if np.any(y_prob < 0) or np.any(y_prob > 1):
    raise ValueError(
        "Predictions are outside the expected probability range."
    )

y_pred = (y_prob >= THRESHOLD).astype(int)


# ============================================================
# STEP 8: CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

auc = roc_auc_score(y_true, y_prob)


# ============================================================
# STEP 9: PRINT EVALUATION RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("MODEL EVALUATION RESULTS")
print("=" * 65)

print(f"\nModel:       {MODEL_PATH.name}")
print(f"Test images: {len(y_true)}")
print(f"Threshold:   {THRESHOLD:.2f}")

print("\nPerformance metrics:")

print(f"Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:    {precision:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"ROC-AUC:      {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Interpretation:")

print(f"True NORMAL:       {tn}")
print(f"False PNEUMONIA:   {fp}")
print(f"Missed PNEUMONIA:  {fn}")
print(f"True PNEUMONIA:    {tp}")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
)


# ============================================================
# STEP 10: SAVE PREDICTIONS TO CSV
# ============================================================

results_df = df.copy()

results_df["predicted_label"] = y_pred

results_df["predicted_class"] = [
    CLASS_NAMES[label]
    for label in y_pred
]

results_df["pneumonia_probability"] = y_prob

results_df["threshold"] = THRESHOLD

results_df["correct_prediction"] = (
    results_df["actual_label"]
    == results_df["predicted_label"]
)

results_df.to_csv(
    PREDICTIONS_CSV,
    index=False,
)

print(f"\nPredictions saved to:\n{PREDICTIONS_CSV}")


# ============================================================
# STEP 11: SAVE METRICS TO JSON
# ============================================================

metrics = {
    "model": MODEL_PATH.name,
    "test_images": int(len(y_true)),
    "threshold": float(THRESHOLD),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "roc_auc": float(auc),
    "true_normal": int(tn),
    "false_pneumonia": int(fp),
    "missed_pneumonia": int(fn),
    "true_pneumonia": int(tp),
    "confusion_matrix": cm.tolist(),
    "class_mapping": CLASS_TO_LABEL,
}

with open(
    METRICS_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )

print(f"Metrics saved to:\n{METRICS_JSON}")


# ============================================================
# STEP 12: SAVE CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

display.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title("DenseNet121 - Test Confusion Matrix")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Confusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PNG}"
)


# ============================================================
# STEP 13: PREDICTION DISTRIBUTION
# ============================================================

print("\nPrediction distribution:")

print(
    pd.Series(
        [
            CLASS_NAMES[label]
            for label in y_pred
        ],
        name="Predicted Class",
    )
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)

print("\nProbability distribution:")

print(f"Minimum probability: {y_prob.min():.6f}")
print(f"Maximum probability: {y_prob.max():.6f}")
print(f"Mean probability:    {y_prob.mean():.6f}")
print(f"Median probability:  {np.median(y_prob):.6f}")


# ============================================================
# STEP 14: COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("VERIFICATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")
print(f"1. {PREDICTIONS_CSV.name}")
print(f"2. {METRICS_JSON.name}")
print(f"3. {CONFUSION_MATRIX_PNG.name}")

print(
    "\nNote: These metrics describe performance on the supplied "
    "test dataset. They do not establish clinical safety."
)
