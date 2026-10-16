import os
import numpy as np
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = r"A:\Project\Medical_Image_Analysis\models\best_model.keras"

DATASET_PATH = r"A:\Project\Medical_Image_Analysis\chest_xray"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# 2. LOAD MODEL
# ==========================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")


# ==========================================
# 3. DISPLAY MODEL ARCHITECTURE
# ==========================================

print("\n========== MODEL SUMMARY ==========\n")

model.summary()

print("\n========== MODEL OUTPUT ==========")

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nFinal layers:")

for layer in model.layers[-5:]:
    print(
        layer.name,
        "|",
        layer.__class__.__name__,
        "|",
        getattr(layer, "activation", None)
    )


# ==========================================
# 4. CHECK DATASET STRUCTURE
# ==========================================

for split in ["train", "val", "test"]:

    split_path = os.path.join(DATASET_PATH, split)

    print(f"\n========== {split.upper()} DATASET ==========")

    if not os.path.isdir(split_path):
        print("Dataset directory not found:", split_path)
        continue

    for class_name in CLASS_NAMES:

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            print(f"{class_name}: DIRECTORY MISSING")
            continue

        image_files = [
            file for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        print(f"{class_name}: {len(image_files)} images")


# ==========================================
# 5. CHECK TRAINING DATA LABEL MAPPING
# ==========================================

train_path = os.path.join(DATASET_PATH, "train")

if os.path.isdir(train_path):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=False
    )

    print("\nTraining dataset class mapping:")
    print(train_ds.class_names)


# ==========================================
# 6. TEST MODEL OUTPUTS
# ==========================================

test_path = os.path.join(DATASET_PATH, "test")

if os.path.isdir(test_path):

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=False
    )

    all_labels = []
    all_scores = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        all_labels.extend(
            labels.numpy().reshape(-1).astype(int)
        )

        all_scores.extend(predictions.reshape(-1))

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    print("\n========== PREDICTION DIAGNOSTICS ==========")

    print("Total images:", len(all_labels))

    print("\nRaw model output shape:", model.output_shape)

    print("\nOverall score statistics:")

    print("Minimum:", np.min(all_scores))
    print("Maximum:", np.max(all_scores))
    print("Mean:", np.mean(all_scores))
    print("Median:", np.median(all_scores))

    for label, class_name in enumerate(CLASS_NAMES):

        class_scores = all_scores[all_labels == label]

        print(f"\n{class_name} images: {len(class_scores)}")

        if len(class_scores) > 0:

            print("Mean score:", np.mean(class_scores))
            print("Minimum score:", np.min(class_scores))
            print("Maximum score:", np.max(class_scores))

            print(
                "Predicted pneumonia at threshold 0.5:",
                np.sum(class_scores >= 0.5),
                "/",
                len(class_scores)
            )

print("\nDiagnostics completed.")
