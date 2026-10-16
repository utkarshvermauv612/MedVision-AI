import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Validation split: 20% of the original training images.
VALIDATION_SIZE = 0.20

# Reproducibility.
tf.keras.utils.set_random_seed(SEED)

print("=" * 65)
print("AI-POWERED MEDICAL IMAGE ANALYSIS")
print("CNN: NORMAL vs PNEUMONIA")
print("=" * 65)


# ============================================================
# STEP 2: VERIFY DATASET DIRECTORIES
# ============================================================

for folder in [TRAIN_DIR, TEST_DIR]:

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            "Please check your chest_xray directory."
        )

    for class_name in CLASS_NAMES:

        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_folder}"
            )

print("\nDataset directories verified successfully.")


# ============================================================
# STEP 3: COLLECT TRAINING IMAGE PATHS AND LABELS
# ============================================================

image_paths = []
image_labels = []

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for class_name in CLASS_NAMES:

    class_folder = TRAIN_DIR / class_name

    image_files = [
        path
        for path in class_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ]

    print(f"{class_name} training images found: {len(image_files)}")

    for image_path in image_files:
        image_paths.append(str(image_path))
        image_labels.append(class_name)


# Create a DataFrame containing image paths and labels.
dataset_df = pd.DataFrame({
    "filepath": image_paths,
    "label": image_labels
})

if dataset_df.empty:
    raise ValueError(
        "No images were found in the training folders."
    )

print("\nTotal training images:", len(dataset_df))

print("\nOriginal training class distribution:")
print(dataset_df["label"].value_counts())


# ============================================================
# STEP 4: CREATE STRATIFIED TRAINING AND VALIDATION SPLITS
# ============================================================

# Use images from the original TRAIN directory only.
# The separate TEST directory is not used during this split.

train_df, val_df = train_test_split(
    dataset_df,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=dataset_df["label"]
)

# Reset DataFrame indices.
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\n" + "=" * 65)
print("TRAINING AND VALIDATION SPLIT")
print("=" * 65)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())


# ============================================================
# STEP 5: CREATE IMAGE DATA GENERATORS
# ============================================================

# Training augmentation is applied only to training images.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.10,
    fill_mode="nearest"
)

# Validation and testing images are only normalized.
eval_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
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
    color_mode="grayscale",
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
    color_mode="grayscale",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# STEP 8: CREATE TEST GENERATOR
# ============================================================

# IMPORTANT:
# The test directory remains separate from training and validation.
# Do not use test images to train or tune the model.

test_generator = eval_datagen.flow_from_directory(
    directory=str(TEST_DIR),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# STEP 9: VERIFY LABELS AND DATASET COUNTS
# ============================================================

print("\n" + "=" * 65)
print("DATASET VERIFICATION")
print("=" * 65)

print("\nClass mapping:")
print(train_generator.class_indices)

print("\nTraining samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)
print("Testing samples:", test_generator.samples)

print("\nExpected class mapping:")
print("NORMAL    = 0")
print("PNEUMONIA = 1")

print("\nTraining generator class counts:")
print(Counter(train_generator.classes))

print("\nValidation generator class counts:")
print(Counter(val_generator.classes))

print("\nTest generator class counts:")
print(Counter(test_generator.classes))

# Confirm that image labels are mapped correctly.
assert train_generator.class_indices == {
    "NORMAL": 0,
    "PNEUMONIA": 1
}, "Unexpected class mapping!"

assert val_generator.class_indices == train_generator.class_indices
assert test_generator.class_indices == train_generator.class_indices

print("\nDataset verification completed successfully.")


# ============================================================
# STEP 10: CALCULATE CLASS WEIGHTS
# ============================================================

# The dataset contains more PNEUMONIA images than NORMAL images.
# Class weights can help the model pay attention to both classes.

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_generator.classes
)

class_weight_dict = {
    0: float(class_weights[0]),
    1: float(class_weights[1])
}

print("\n" + "=" * 65)
print("CLASS WEIGHTS")
print("=" * 65)

print("NORMAL weight:", class_weight_dict[0])
print("PNEUMONIA weight:", class_weight_dict[1])


# ============================================================
# STEP 11: BUILD THE CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(shape=(256, 256, 1)),

    # Convolution block 1
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolution block 2
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolution block 3
    layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolution block 4
    layers.Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Feature extraction
    layers.GlobalAveragePooling2D(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Reduce overfitting
    layers.Dropout(0.5),

    # Binary output:
    # Lower probability -> NORMAL
    # Higher probability -> PNEUMONIA
    layers.Dense(1, activation="sigmoid")
])


# ============================================================
# STEP 12: COMPILE THE CNN MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\n" + "=" * 65)
print("CNN MODEL ARCHITECTURE")
print("=" * 65)

model.summary()


# ============================================================
# STEP 13: SET UP TRAINING CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "best_model_v2.keras"

callbacks = [

    # Stop training when validation loss stops improving.
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Save the best checkpoint.
    ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    # Reduce learning rate when progress stalls.
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# STEP 14: TRAIN THE CNN MODEL
# ============================================================

print("\n" + "=" * 65)
print("STARTING CNN TRAINING")
print("=" * 65)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


# ============================================================
# STEP 15: DISPLAY FINAL TRAINING METRICS
# ============================================================

print("\n" + "=" * 65)
print("FINAL TRAINING METRICS")
print("=" * 65)

for metric_name, values in history.history.items():

    if values:
        print(
            f"{metric_name}: {values[-1]:.4f}"
        )


# ============================================================
# STEP 16: SAVE THE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "final_model_v2.keras"

model.save(str(final_model_path))

print("\nTraining completed.")

print("Final model saved at:")
print(final_model_path)

print("\nBest checkpoint saved at:")
print(best_model_path)


# ============================================================
# STEP 17: PLOT TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_plot_path = MODEL_DIR / "accuracy_v2.png"

plt.savefig(
    str(accuracy_plot_path),
    dpi=150
)

plt.show()

print("\nAccuracy graph saved at:")
print(accuracy_plot_path)


# ============================================================
# STEP 18: PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = MODEL_DIR / "loss_v2.png"

plt.savefig(
    str(loss_plot_path),
    dpi=150
)

plt.show()

print("\nLoss graph saved at:")
print(loss_plot_path)


# ============================================================
# STEP 19: LOAD THE BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 65)
print("LOADING BEST MODEL CHECKPOINT")
print("=" * 65)

if not best_model_path.exists():
    raise FileNotFoundError(
        "Best model checkpoint was not created. "
        "Please check the training output for errors."
    )

best_model = tf.keras.models.load_model(
    str(best_model_path)
)

print("\nBest model loaded successfully.")

print("\nModel input shape:")
print(best_model.input_shape)

print("\nModel output shape:")
print(best_model.output_shape)


# ============================================================
# STEP 20: EVALUATE ON THE SEPARATE TEST DATASET
# ============================================================

print("\n" + "=" * 65)
print("EVALUATING MODEL ON TEST DATASET")
print("=" * 65)

test_generator.reset()

test_results = best_model.evaluate(
    test_generator,
    return_dict=True,
    verbose=1
)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

for metric_name, value in test_results.items():
    print(f"{metric_name}: {value:.4f}")


# ============================================================
# STEP 21: GENERATE CLASSIFICATION REPORT
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

print("\nGenerating test predictions...")

test_generator.reset()

predicted_probabilities = best_model.predict(
    test_generator,
    verbose=1
).ravel()

# Use 0.5 as a baseline threshold.
# This threshold should be assessed and may need adjustment
# based on validation performance and project requirements.
predicted_labels = (
    predicted_probabilities >= 0.5
).astype(int)

true_labels = test_generator.classes

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4
    )
)


# ============================================================
# STEP 22: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    values_format="d"
)

plt.title("CNN Test Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = MODEL_DIR / "confusion_matrix_v2.png"

plt.savefig(
    str(confusion_matrix_path),
    dpi=150
)

plt.show()

print("\nConfusion matrix saved at:")
print(confusion_matrix_path)


# ============================================================
# STEP 23: CALCULATE SENSITIVITY AND SPECIFICITY
# ============================================================

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

print("\n" + "=" * 65)
print("ADDITIONAL TEST METRICS")
print("=" * 65)

print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

print(f"\nSensitivity / Recall: {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")

if len(np.unique(true_labels)) == 2:

    test_auc = roc_auc_score(
        true_labels,
        predicted_probabilities
    )

    print(f"ROC-AUC:              {test_auc:.4f}")

else:
    print("ROC-AUC could not be calculated because the test set contains only one class.")


# ============================================================
# STEP 24: SAVE TEST PREDICTIONS
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

    "pneumonia_probability": predicted_probabilities

})

predictions_path = MODEL_DIR / "test_predictions_v2.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("\nTest predictions saved at:")
print(predictions_path)


# ============================================================
# STEP 25: COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print("1. Best model:")
print(best_model_path)

print("\n2. Final model:")
print(final_model_path)

print("\n3. Accuracy graph:")
print(accuracy_plot_path)

print("\n4. Loss graph:")
print(loss_plot_path)

print("\n5. Confusion matrix:")
print(confusion_matrix_path)

print("\n6. Test predictions CSV:")
print(predictions_path)

print("\nIMPORTANT:")
print(
    "This model is for educational and research purposes only. "
    "It must not be used as a substitute for professional medical diagnosis."
)
