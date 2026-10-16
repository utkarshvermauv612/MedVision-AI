from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


# ==========================================
# STEP 1: FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

OLD_CSV = BASE_DIR / "models" / "test_predictions_v3.csv"
NEW_CSV = BASE_DIR / "models" / "verification_predictions_v3.csv"

OUTPUT_CSV = BASE_DIR / "models" / "prediction_score_comparison.csv"


# ==========================================
# STEP 2: LOAD CSV FILES
# ==========================================

print("Loading prediction files...")

if not OLD_CSV.exists():
    raise FileNotFoundError(f"Previous CSV not found: {OLD_CSV}")

if not NEW_CSV.exists():
    raise FileNotFoundError(f"Current CSV not found: {NEW_CSV}")

old = pd.read_csv(OLD_CSV)
new = pd.read_csv(NEW_CSV)

required_columns = {
    "filepath",
    "actual_label",
    "predicted_label",
    "pneumonia_probability"
}

for name, dataframe in [("Previous", old), ("Current", new)]:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {sorted(missing)}"
        )

print("Previous predictions:", len(old))
print("Current predictions:", len(new))


# ==========================================
# STEP 3: NORMALIZE FILENAMES
# ==========================================

old["filename"] = old["filepath"].apply(
    lambda path: Path(str(path).replace("\\", "/")).name.lower()
)

new["filename"] = new["filepath"].apply(
    lambda path: Path(str(path).replace("\\", "/")).name.lower()
)

old["actual_label"] = old["actual_label"].astype(str).str.upper()
new["actual_label"] = new["actual_label"].astype(str).str.upper()

old["predicted_label"] = old["predicted_label"].astype(str).str.upper()
new["predicted_label"] = new["predicted_label"].astype(str).str.upper()


# ==========================================
# STEP 4: RENAME SCORE COLUMNS BEFORE MERGING
# ==========================================

old = old.rename(
    columns={
        "pneumonia_probability": "pneumonia_probability_old",
        "actual_label": "actual_label_old",
        "predicted_label": "predicted_label_old"
    }
)

new = new.rename(
    columns={
        "pneumonia_probability": "pneumonia_probability_new",
        "actual_label": "actual_label_new",
        "predicted_label": "predicted_label_new"
    }
)


# ==========================================
# STEP 5: MERGE AND VALIDATE
# ==========================================

merged = old[
    [
        "filename",
        "actual_label_old",
        "predicted_label_old",
        "pneumonia_probability_old"
    ]
].merge(
    new[
        [
            "filename",
            "actual_label_new",
            "predicted_label_new",
            "pneumonia_probability_new"
        ]
    ],
    on="filename",
    how="outer",
    validate="one_to_one",
    indicator=True
)

print("\nImage matching:")
print(merged["_merge"].value_counts())

if not (merged["_merge"] == "both").all():
    raise ValueError(
        "The files do not contain exactly the same image filenames."
    )

if not (
    merged["actual_label_old"] == merged["actual_label_new"]
).all():
    raise ValueError(
        "Actual labels differ between the prediction files."
    )


# ==========================================
# STEP 6: COMPARE PREDICTION SCORES
# ==========================================

old_scores = merged["pneumonia_probability_old"]
new_scores = merged["pneumonia_probability_new"]

merged["absolute_score_difference"] = (
    old_scores - new_scores
).abs()

old_accuracy = accuracy_score(
    merged["actual_label_old"],
    merged["predicted_label_old"]
)

new_accuracy = accuracy_score(
    merged["actual_label_new"],
    merged["predicted_label_new"]
)

print("\n" + "=" * 60)
print("PREDICTION SCORE COMPARISON")
print("=" * 60)

print(f"Images compared: {len(merged)}")
print(f"Previous accuracy: {old_accuracy * 100:.2f}%")
print(f"Current accuracy: {new_accuracy * 100:.2f}%")

print(
    "Mean absolute score difference:",
    f"{merged['absolute_score_difference'].mean():.6f}"
)

print(
    "Maximum absolute score difference:",
    f"{merged['absolute_score_difference'].max():.6f}"
)

nearly_identical = (
    merged["absolute_score_difference"] <= 0.0001
).sum()

print(
    f"Nearly identical scores: {nearly_identical} "
    f"of {len(merged)}"
)


# ==========================================
# STEP 7: FIND CHANGED PREDICTIONS
# ==========================================

changed = merged[
    merged["predicted_label_old"]
    != merged["predicted_label_new"]
]

print(f"\nChanged predictions: {len(changed)}")

if not changed.empty:
    print("\nFirst 15 changed predictions:")

    print(
        changed[
            [
                "filename",
                "actual_label_old",
                "pneumonia_probability_old",
                "predicted_label_old",
                "pneumonia_probability_new",
                "predicted_label_new",
                "absolute_score_difference"
            ]
        ].head(15).to_string(index=False)
    )


# ==========================================
# STEP 8: SAVE RESULTS
# ==========================================

merged.drop(columns=["_merge"]).to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nComparison saved to:")
print(OUTPUT_CSV)

print("\nScore comparison completed successfully.")