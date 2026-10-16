from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "models" / "test_predictions_v3.csv"

OUTPUT_PATH = BASE_DIR / "models" / "threshold_analysis_v3.csv"


# ==========================================
# STEP 2: LOAD CSV
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())


# ==========================================
# STEP 3: CONVERT LABELS TO NUMBERS
# ==========================================

# NORMAL = 0
# PNEUMONIA = 1

label_mapping = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

df["actual_label"] = (
    df["actual_label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

if not set(df["actual_label"].unique()).issubset(label_mapping):
    raise ValueError(
        "Unexpected actual_label values: "
        f"{df['actual_label'].unique()}"
    )

y_true = df["actual_label"].map(label_mapping).to_numpy()

y_probability = pd.to_numeric(
    df["pneumonia_probability"],
    errors="raise"
).to_numpy(dtype=float)

if not np.isfinite(y_probability).all():
    raise ValueError("Invalid probability values found.")

if ((y_probability < 0) | (y_probability > 1)).any():
    raise ValueError("Probabilities must be between 0 and 1.")


# ==========================================
# STEP 4: CALCULATE ROC-AUC
# ==========================================

auc = roc_auc_score(y_true, y_probability)


# ==========================================
# STEP 5: ANALYSE THRESHOLDS
# ==========================================

thresholds = np.arange(0.20, 0.81, 0.05)

results = []

for threshold in thresholds:

    # Probability >= threshold means PNEUMONIA
    # Probability < threshold means NORMAL

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ==========================================
# STEP 6: DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("VERSION 3 DENSENET121 THRESHOLD ANALYSIS")
print("=" * 100)

print(
    results_df.round(4).to_string(index=False)
)

print("\nROC-AUC:", round(auc, 4))


# ==========================================
# STEP 7: SAVE RESULTS
# ==========================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nAnalysis successfully saved to:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "These thresholds were evaluated on the test set. "
    "Use these results for exploratory analysis only. "
    "Choose the final threshold using validation data."
)