from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================================
# STEP 1: CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

V2_CSV = MODELS_DIR / "test_predictions_v2.csv"
V3_CSV = MODELS_DIR / "test_predictions_v3.csv"

OUTPUT_CSV = MODELS_DIR / "model_performance_comparison.csv"
OUTPUT_JSON = MODELS_DIR / "model_performance_comparison.json"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Both saved prediction files were generated with
# a probability threshold of 0.50.
THRESHOLD = 0.50


# =====================================================
# STEP 2: LOAD AND VALIDATE PREDICTION FILES
# =====================================================

def load_predictions(csv_path, model_name):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "actual_label",
        "pneumonia_probability"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} CSV is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["actual_label"] = (
        df["actual_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed_labels = {"NORMAL", "PNEUMONIA"}

    if not set(df["actual_label"].unique()).issubset(allowed_labels):
        raise ValueError(
            f"Unexpected labels in {model_name}: "
            f"{df['actual_label'].unique()}"
        )

    df["pneumonia_probability"] = pd.to_numeric(
        df["pneumonia_probability"],
        errors="raise"
    )

    if not np.isfinite(df["pneumonia_probability"]).all():
        raise ValueError(
            f"{model_name} contains invalid probabilities."
        )

    if (
        (df["pneumonia_probability"] < 0) |
        (df["pneumonia_probability"] > 1)
    ).any():
        raise ValueError(
            f"{model_name} probabilities must be between 0 and 1."
        )

    return df


# =====================================================
# STEP 3: CALCULATE METRICS
# =====================================================

def evaluate_model(df, model_name):

    y_true = (
        df["actual_label"]
        .map({"NORMAL": 0, "PNEUMONIA": 1})
        .to_numpy()
    )

    probabilities = df[
        "pneumonia_probability"
    ].to_numpy(dtype=float)

    y_pred = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "model": model_name,
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print("\n" + "=" * 65)
    print(f"{model_name} — EVALUATION")
    print("=" * 65)

    for key, value in metrics.items():
        if key != "model":
            if isinstance(value, float):
                print(f"{key:18s}: {value:.4f}")
            else:
                print(f"{key:18s}: {value}")

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

    return metrics, y_true, y_pred, probabilities


# =====================================================
# STEP 4: LOAD BOTH MODELS' SAVED PREDICTIONS
# =====================================================

print("Loading saved predictions...")

df_v2 = load_predictions(V2_CSV, "Version 2 CNN")
df_v3 = load_predictions(V3_CSV, "Version 3 DenseNet121")

print(f"Version 2 images: {len(df_v2)}")
print(f"Version 3 images: {len(df_v3)}")


# =====================================================
# STEP 5: CHECK THAT THE TEST IMAGES MATCH
# =====================================================

if "filepath" in df_v2.columns and "filepath" in df_v3.columns:

    paths_v2 = set(df_v2["filepath"].astype(str))
    paths_v3 = set(df_v3["filepath"].astype(str))

    if paths_v2 != paths_v3:
        raise ValueError(
            "The two CSV files do not contain the same test image paths. "
            "A direct model comparison would be misleading."
        )

    # Align rows in case the prediction CSVs use different orders.
    df_v3 = (
        df_v3.set_index("filepath")
        .loc[df_v2["filepath"].astype(str)]
        .reset_index()
    )

if len(df_v2) != len(df_v3):
    raise ValueError(
        "The prediction files contain different numbers of images."
    )

if not np.array_equal(
    df_v2["actual_label"].to_numpy(),
    df_v3["actual_label"].to_numpy()
):
    raise ValueError(
        "Actual labels differ between the prediction files."
    )


# =====================================================
# STEP 6: EVALUATE BOTH MODELS
# =====================================================

metrics_v2, y2, pred2, prob2 = evaluate_model(
    df_v2,
    "Version 2 — Custom CNN"
)

metrics_v3, y3, pred3, prob3 = evaluate_model(
    df_v3,
    "Version 3 — DenseNet121"
)


# =====================================================
# STEP 7: SAVE COMPARISON TABLE
# =====================================================

comparison_df = pd.DataFrame([metrics_v2, metrics_v3])

comparison_df.to_csv(
    OUTPUT_CSV,
    index=False
)

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(
        [metrics_v2, metrics_v3],
        file,
        indent=4
    )

print("\n" + "=" * 85)
print("VERSION 2 vs VERSION 3")
print("=" * 85)

display_columns = [
    "model",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "roc_auc"
]

print(
    comparison_df[display_columns]
    .round(4)
    .to_string(index=False)
)


# =====================================================
# STEP 8: GENERATE CONFUSION MATRIX COMPARISON
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y2,
    pred2,
    labels=[0, 1],
    display_labels=CLASS_NAMES,
    cmap="Blues",
    ax=axes[0],
    colorbar=False
)

axes[0].set_title("Version 2 — Custom CNN")

ConfusionMatrixDisplay.from_predictions(
    y3,
    pred3,
    labels=[0, 1],
    display_labels=CLASS_NAMES,
    cmap="Greens",
    ax=axes[1],
    colorbar=False
)

axes[1].set_title("Version 3 — DenseNet121")

fig.suptitle("Confusion Matrix Comparison")
fig.tight_layout()

confusion_path = MODELS_DIR / "confusion_matrix_comparison.png"

fig.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================================
# STEP 9: GENERATE METRIC COMPARISON CHART
# =====================================================

chart_metrics = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1_score",
    "roc_auc"
]

x = np.arange(len(chart_metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width / 2,
    [metrics_v2[m] for m in chart_metrics],
    width,
    label="Version 2 — Custom CNN"
)

ax.bar(
    x + width / 2,
    [metrics_v3[m] for m in chart_metrics],
    width,
    label="Version 3 — DenseNet121"
)

ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(chart_metrics, rotation=20)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

chart_path = MODELS_DIR / "model_performance_comparison.png"

fig.savefig(
    chart_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================================
# STEP 10: FINISH
# =====================================================

print("\n" + "=" * 65)
print("COMPARISON COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print("CSV:", OUTPUT_CSV)
print("JSON:", OUTPUT_JSON)
print("Confusion matrices:", confusion_path)
print("Performance chart:", chart_path)

print(
    "\nIMPORTANT: These are previously examined test-set results. "
    "Treat this comparison as exploratory. Do not use it to choose "
    "the final model or threshold and then report the same test set "
    "as an independent evaluation."
)