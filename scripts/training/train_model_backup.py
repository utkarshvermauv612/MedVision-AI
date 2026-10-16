import os
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "chest_xray" / "train"
VAL_DIR = BASE_DIR / "chest_xray" / "val"
TEST_DIR = BASE_DIR / "chest_xray" / "test"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# STEP 2: CHECK DATASET DIRECTORIES
# ==========================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

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

print("Dataset directories verified successfully.")


# ==========================================
# STEP 3: IMAGE PREPROCESSING
# ==========================================

# Augmentation is applied only to training images.
# Horizontal flipping is disabled to preserve
# left/right anatomical orientation.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.10,
    fill_mode="nearest"
)

# Validation and test images are not augmented.

eval_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ==========================================
# STEP 4: LOAD TRAINING DATA
# ==========================================

train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\n===== CLASS LABEL CHECK =====")
print(train_generator.class_indices)
print(train_generator.classes[:20])


# ==========================================
# STEP 5: LOAD VALIDATION DATA
# ==========================================

val_generator = eval_datagen.flow_from_directory(
    str(VAL_DIR),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

from collections import Counter

print("\n===== TRAINING DATA CHECK =====")
print("Training class mapping:", train_generator.class_indices)
print("Training images:", train_generator.samples)
print("Training class counts:", Counter(train_generator.classes))

print("\n===== VALIDATION DATA CHECK =====")
print("Validation class mapping:", val_generator.class_indices)
print("Validation images:", val_generator.samples)
print("Validation class counts:", Counter(val_generator.classes))


# ==========================================
# STEP 6: LOAD TEST DATA
# ==========================================

test_generator = eval_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass mapping:")
print(train_generator.class_indices)

print("\nTraining images:", train_generator.samples)
print("Validation images:", val_generator.samples)
print("Testing images:", test_generator.samples)


# ==========================================
# STEP 7: BUILD CNN ARCHITECTURE
# ==========================================

model = models.Sequential([

    # Input: 256 x 256 grayscale image
    layers.Input(shape=(256, 256, 1)),

    # Convolutional Block 1
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 2
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 3
    layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 4
    layers.Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    # Reduce spatial dimensions
    layers.GlobalAveragePooling2D(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.5),

    # Binary classification output
    # 0 = NORMAL, 1 = PNEUMONIA
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# STEP 8: COMPILE THE CNN MODEL
# ==========================================

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

print("\nCNN Model Architecture:")
model.summary()


# ==========================================
# STEP 9: CONFIGURE CALLBACKS
# ==========================================

callbacks = [

    # Stop training when validation loss stops improving
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    # Save the best model
    ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when progress plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ==========================================
# STEP 10: TRAIN THE CNN
# ==========================================

print("\nStarting CNN training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n===== FINAL TRAINING METRICS =====")

for metric, values in history.history.items():
    print(f"{metric}: {values[-1]:.4f}")

print("\n===== FINAL EPOCH =====")

for key, values in history.history.items():
    print(f"{key}: {values[-1]:.4f}")


# ==========================================
# STEP 11: SAVE THE TRAINED MODEL
# ==========================================

final_model_path = MODEL_DIR / "final_model.keras"

model.save(str(final_model_path))

print("\nTraining completed!")
print(f"Final model saved at: {final_model_path}")

print(
    f"Best checkpoint saved at: "
    f"{MODEL_DIR / 'best_model.keras'}"
)


# ==========================================
# STEP 12: PLOT ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

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
plt.show()


# ==========================================
# STEP 13: PLOT LOSS
# ==========================================

plt.figure(figsize=(8, 5))

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
plt.show()


# ==========================================
# STEP 14: EVALUATE THE BEST MODEL
# ==========================================

best_model = tf.keras.models.load_model(
    str(MODEL_DIR / "best_model.keras")
)

print("\nEvaluating model on test dataset...")

test_results = best_model.evaluate(
    test_generator,
    return_dict=True,
    verbose=1
)

print("\n========== TEST RESULTS ==========")

for metric, value in test_results.items():

    print(f"{metric}: {value:.4f}")

print("\nCNN training and evaluation completed.")