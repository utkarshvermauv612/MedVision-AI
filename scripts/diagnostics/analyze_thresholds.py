from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# =====================================================
# STEP 1: LOAD EXISTING TEST PREDICTIONS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "models"
    / "test_predictions_v2.csv"
)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print("Total images:", len(df))

# NORMAL = 0
# PNEUMONIA = 1

y_true = (
    df["actual_label"]
    .map({
        "NORMAL": 0,
        "PNEUMONIA": 1
    })
    .to_numpy()
)

probabilities = (
    df["pneumonia_probability"]
    .to_numpy()
)

if np.isnan(y_true).any():
    raise ValueError("Unexpected actual class label.")

if np.isnan(probabilities).any():
    raise ValueError("Missing prediction probabilities.")

# =====================================================
# STEP 2: REPORT ROC-AUC
# =====================================================

print("\nROC-AUC:")
print(f"{roc_auc_score(y_true, probabilities):.4f}")

# =====================================================
# STEP 3: COMPARE CLASSIFICATION THRESHOLDS
# =====================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

results = []

for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

results_df = pd.DataFrame(results)

print("\nResults at different thresholds:\n")

print(
    results_df.round(4).to_string(index=False)
)

# =====================================================
# STEP 4: SAVE RESULTS
# =====================================================

output_path = (
    BASE_DIR
    / "models"
    / "threshold_analysis_v2.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)

print("\nAnalysis completed.")