from pathlib import Path

import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# 1. Check saved model files
# --------------------------------------------------

print("=" * 60)
print("SAVED MODEL FILES")
print("=" * 60)

for filename in [
    "best_model_v2.keras",
    "final_model_v2.keras",
    "accuracy_v2.png",
    "loss_v2.png",
    "confusion_matrix_v2.png",
    "test_predictions_v2.csv"
]:
    path = MODEL_DIR / filename
    print(f"{filename}: {'EXISTS' if path.exists() else 'NOT FOUND'}")


# --------------------------------------------------
# 2. Inspect the best model
# --------------------------------------------------

model_path = MODEL_DIR / "best_model_v2.keras"

if model_path.exists():

    model = tf.keras.models.load_model(str(model_path))

    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Total parameters:", model.count_params())

    print("\nModel layers:")

    for layer in model.layers:
        print(
            f"{layer.name}: "
            f"{layer.__class__.__name__}, "
            f"output shape={layer.output.shape}"
        )


# --------------------------------------------------
# 3. Analyze existing test predictions
# --------------------------------------------------

predictions_path = MODEL_DIR / "test_predictions_v2.csv"

if predictions_path.exists():

    df = pd.read_csv(predictions_path)

    print("\n" + "=" * 60)
    print("TEST PREDICTION ANALYSIS")
    print("=" * 60)

    print("\nTotal test images:", len(df))

    print("\nActual class counts:")
    print(df["actual_label"].value_counts())

    print("\nPredicted class counts:")
    print(df["predicted_label"].value_counts())

    print("\nPrediction correctness:")

    df["correct"] = (
        df["actual_label"] == df["predicted_label"]
    )

    print(df["correct"].value_counts())

    print("\nCorrect predictions by actual class:")

    for label in ["NORMAL", "PNEUMONIA"]:

        subset = df[df["actual_label"] == label]

        if len(subset) > 0:

            accuracy = subset["correct"].mean()

            print(
                f"{label}: "
                f"{accuracy:.4f} "
                f"({subset['correct'].sum()}/{len(subset)})"
            )

    print("\nPneumonia probability statistics:")

    print(
        df["pneumonia_probability"].describe().to_string()
    )

    print("\nMean pneumonia probability by actual class:")

    print(
        df.groupby("actual_label")[
            "pneumonia_probability"
        ].mean().to_string()
    )

    print("\nSample predictions:")

    print(df.head(10).to_string(index=False))

else:

    print("\nTest predictions CSV was not found.")


print("\nAnalysis completed.")