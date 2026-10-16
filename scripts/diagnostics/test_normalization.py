import numpy as np
import tensorflow as tf

MODEL_PATH = r"A:\Project\Medical_Image_Analysis\models\best_model.keras"

TEST_DIR = r"A:\Project\Medical_Image_Analysis\chest_xray\test"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    color_mode="grayscale",
    image_size=(256, 256),
    batch_size=32,
    shuffle=False
)

# Collect predictions for both preprocessing methods
raw_scores = []
normalized_scores = []
actual_labels = []

for images, labels in test_ds:

    images = tf.cast(images, tf.float32)

    # Method 1: Original pixel values (0-255)
    raw_predictions = model(images, training=False)

    # Method 2: Normalized pixel values (0-1)
    normalized_images = images / 255.0

    normalized_predictions = model(
        normalized_images,
        training=False
    )

    raw_scores.extend(
        raw_predictions.numpy().reshape(-1)
    )

    normalized_scores.extend(
        normalized_predictions.numpy().reshape(-1)
    )

    actual_labels.extend(
        labels.numpy().reshape(-1).astype(int)
    )

raw_scores = np.array(raw_scores)
normalized_scores = np.array(normalized_scores)
actual_labels = np.array(actual_labels)

print("\n========== RAW IMAGE RESULTS ==========")

print("Minimum:", raw_scores.min())
print("Maximum:", raw_scores.max())
print("Mean:", raw_scores.mean())
print("Predicted NORMAL:", np.sum(raw_scores < 0.5))
print("Predicted PNEUMONIA:", np.sum(raw_scores >= 0.5))

print("\n========== NORMALIZED IMAGE RESULTS ==========")

print("Minimum:", normalized_scores.min())
print("Maximum:", normalized_scores.max())
print("Mean:", normalized_scores.mean())
print("Predicted NORMAL:", np.sum(normalized_scores < 0.5))
print("Predicted PNEUMONIA:", np.sum(normalized_scores >= 0.5))

print("\n========== CLASS-WISE RESULTS ==========")

for class_id, class_name in enumerate(["NORMAL", "PNEUMONIA"]):

    mask = actual_labels == class_id

    print(f"\n{class_name}")

    print(
        "Raw mean score:",
        raw_scores[mask].mean()
    )

    print(
        "Normalized mean score:",
        normalized_scores[mask].mean()
    )

print("\nNormalization comparison completed.")