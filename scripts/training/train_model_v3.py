import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "chest_xray" / "train"
TEST_DIR = BASE_DIR / "chest_xray" / "test"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8

SEED = 42
VALIDATION_SIZE = 0.20

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

tf.keras.utils.set_random_seed(SEED)

print("=" * 70)
print("AI-POWERED MEDICAL IMAGE ANALYSIS")
print("VERSION 3: DENSENET121 TRANSFER LEARNING")
print("=" * 70)


# ============================================================
# STEP 2: VERIFY DATASET
# ============================================================

for folder in [TRAIN_DIR, TEST_DIR]:

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for class_name in CLASS_NAMES:

        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_folder}"
            )

print("\nDataset directories verified.")


# ============================================================
# STEP 3: COLLECT TRAINING IMAGE PATHS
# ============================================================

valid_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

image_paths = []
image_labels = []

for class_name in CLASS_NAMES:

    class_folder = TRAIN_DIR / class_name

    image_files = sorted([
        path
        for path in class_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ])

    print(
        f"{class_name} images: {len(image_files)}"
    )

    for image_path in image_files:

        image_paths.append(str(image_path))
        image_labels.append(class_name)


dataset_df = pd.DataFrame({
    "filepath": image_paths,
    "label": image_labels
})

if dataset_df.empty:
    raise ValueError(
        "No training images were found."
    )

print("\nTotal training images:", len(dataset_df))

print("\nClass distribution:")
print(dataset_df["label"].value_counts())


# ============================================================
# STEP 4: CREATE TRAINING AND VALIDATION SPLITS
# ============================================================

# Split the original training directory into:
# 80% training
# 20% validation
#
# The separate test directory is not included in this split.

train_df, val_df = train_test_split(
    dataset_df,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=dataset_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())


# ============================================================
# STEP 5: DENSENET121 PREPROCESSING
# ============================================================

# DenseNet121 pretrained weights expect RGB images.
#
# Its preprocessing function handles the expected input scaling.
# Therefore, DO NOT add rescale=1/255 to these generators.

preprocess_function = (
    tf.keras.applications.densenet.preprocess_input
)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rotation_range=8,
    zoom_range=0.08,
    width_shift_range=0.03,
    height_shift_range=0.03,
    fill_mode="nearest"
)

eval_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function
)


# ============================================================
# STEP 6: CREATE TRAINING GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    classes=CLASS_NAMES,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# STEP 7: CREATE VALIDATION GENERATOR
# ============================================================

val_generator = eval_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="label",
    classes=CLASS_NAMES,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# STEP 8: CREATE TEST GENERATOR
# ============================================================

# Do not use the test images for training or validation.

test_generator = eval_datagen.flow_from_directory(
    directory=str(TEST_DIR),
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# STEP 9: VERIFY DATASET AND LABELS
# ============================================================

print("\n" + "=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

print("Class mapping:", train_generator.class_indices)

print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)
print("Test samples:", test_generator.samples)

assert train_generator.class_indices == {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

assert (
    val_generator.class_indices
    == train_generator.class_indices
)

assert (
    test_generator.class_indices
    == train_generator.class_indices
)

print("\nDataset verification completed.")


# ============================================================
# STEP 10: CALCULATE CLASS WEIGHTS
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_generator.classes
)

class_weight_dict = {
    0: float(class_weights[0]),
    1: float(class_weights[1])
}

print("\nClass weights:")
print(class_weight_dict)


# ============================================================
# STEP 11: LOAD PRETRAINED DENSENET121
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED DENSENET121")
print("=" * 70)

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Initially freeze the pretrained feature extractor.
base_model.trainable = False

print("\nDenseNet121 loaded successfully.")


# ============================================================
# STEP 12: BUILD THE CLASSIFICATION MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="chest_xray_input"
)

# Extract features using DenseNet121.
x = base_model(
    inputs,
    training=False
)

# Convert feature maps into a feature vector.
x = layers.GlobalAveragePooling2D()(x)

# Regularization.
x = layers.Dropout(0.35)(x)

# Classification layer.
outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="pneumonia_probability"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="DenseNet121_ChestXray"
)


# ============================================================
# STEP 13: COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nModel summary:")
model.summary()


# ============================================================
# STEP 14: SET UP CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR / "best_model_v3.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# STEP 15: INITIAL TRANSFER-LEARNING TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 70)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


# ============================================================
# STEP 16: FINE-TUNE THE LAST DENSENET LAYERS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING")
print("=" * 70)

# Unfreeze the base model, then freeze most of its layers.
# Only the last portion will be fine-tuned.

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

# Fine-tune only the later layers.
for layer in base_model.layers[-40:]:

    # Keep BatchNormalization layers frozen for stability.
    if not isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = True


# Recompile after changing trainable layers.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)

# Continue epoch numbering after Phase 1.
initial_completed = len(
    history_initial.history["loss"]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    initial_epoch=initial_completed,
    epochs=initial_completed + FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


# ============================================================
# STEP 17: COMBINE TRAINING HISTORY
# ============================================================

combined_history = {}

for key in history_initial.history:

    combined_history[key] = (
        history_initial.history[key]
        + history_fine.history[key]
    )

history_df = pd.DataFrame(
    combined_history
)

history_df.insert(
    0,
    "epoch",
    range(1, len(history_df) + 1)
)

history_csv_path = (
    MODEL_DIR / "training_history_v3.csv"
)

history_df.to_csv(
    history_csv_path,
    index=False
)

print("\nTraining history saved:")
print(history_csv_path)

print("\nTraining history:")
print(history_df.to_string(index=False))


# ============================================================
# STEP 18: SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR / "final_model_v3.keras"
)

model.save(str(final_model_path))

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# STEP 19: PLOT ACCURACY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history_df["epoch"],
    history_df["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("DenseNet121 Training and Validation Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_path = (
    MODEL_DIR / "accuracy_v3.png"
)

plt.savefig(
    str(accuracy_path),
    dpi=150
)

plt.close()


# ============================================================
# STEP 20: PLOT LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history_df["epoch"],
    history_df["loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("DenseNet121 Training and Validation Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

loss_path = (
    MODEL_DIR / "loss_v3.png"
)

plt.savefig(
    str(loss_path),
    dpi=150
)

plt.close()

print("\nAccuracy graph saved:", accuracy_path)
print("Loss graph saved:", loss_path)


# ============================================================
# STEP 21: LOAD THE BEST CHECKPOINT
# ============================================================

if not best_model_path.exists():
    raise FileNotFoundError(
        "Best checkpoint was not created."
    )

best_model = tf.keras.models.load_model(
    str(best_model_path)
)

print("\nBest checkpoint loaded successfully.")


# ============================================================
# STEP 22: EVALUATE THE TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_generator.reset()

test_results = best_model.evaluate(
    test_generator,
    return_dict=True,
    verbose=1
)

print("\nTest metrics:")

for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")


# ============================================================
# STEP 23: GENERATE TEST PREDICTIONS
# ============================================================

test_generator.reset()

probabilities = best_model.predict(
    test_generator,
    verbose=1
).ravel()

# Baseline threshold. Do not tune it using test results.
predicted_labels = (
    probabilities >= 0.5
).astype(int)

true_labels = test_generator.classes


# ============================================================
# STEP 24: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# STEP 25: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)

print("\nConfusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

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

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

if len(np.unique(true_labels)) == 2:

    test_auc = roc_auc_score(
        true_labels,
        probabilities
    )

    print(f"ROC-AUC: {test_auc:.4f}")


cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

cm_display.plot(
    values_format="d"
)

plt.title("DenseNet121 Test Confusion Matrix")
plt.tight_layout()

cm_path = (
    MODEL_DIR / "confusion_matrix_v3.png"
)

plt.savefig(
    str(cm_path),
    dpi=150
)

plt.close()

print("\nConfusion matrix saved:", cm_path)


# ============================================================
# STEP 26: SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "filepath": test_generator.filepaths,

    "actual_label": [
        CLASS_NAMES[label]
        for label in true_labels
    ],

    "predicted_label": [
        CLASS_NAMES[label]
        for label in predicted_labels
    ],

    "pneumonia_probability": probabilities
})

predictions_path = (
    MODEL_DIR / "test_predictions_v3.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("\nTest predictions saved:")
print(predictions_path)


# ============================================================
# STEP 27: SAVE METRICS SUMMARY
# ============================================================

summary = {
    "model": "DenseNet121",
    "input_size": "224x224x3",
    "test_images": int(len(true_labels)),
    "accuracy": float(
        accuracy_score(true_labels, predicted_labels)
    ),
    "precision": float(
        precision_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            true_labels,
            predicted_labels,
            zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            true_labels,
            probabilities
        )
    ),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp)
}

summary_path = (
    MODEL_DIR / "metrics_v3.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print("\nMetrics summary saved:")
print(summary_path)


# ============================================================
# STEP 28: COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DENSENET121 TRAINING COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("Best model:", best_model_path)
print("Final model:", final_model_path)
print("Training history:", history_csv_path)
print("Accuracy graph:", accuracy_path)
print("Loss graph:", loss_path)
print("Confusion matrix:", cm_path)
print("Test predictions:", predictions_path)
print("Metrics summary:", summary_path)

print("\nIMPORTANT:")
print(
    "This model is intended for educational and research "
    "purposes only, not clinical diagnosis."
)