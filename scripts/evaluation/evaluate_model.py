import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = BASE_DIR / "chest_xray" / "test"

MODEL_PATH = BASE_DIR / "models" / "best_model_v3.keras"

IMAGE_SIZE = (256, 256)

BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# STEP 2: LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH)
)

print("Model loaded successfully!")


# ==========================================
# STEP 3: LOAD TEST DATASET
# ==========================================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=CLASS_NAMES,
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass mapping:")
print(test_generator.class_indices)

print("Total test images:", test_generator.samples)


# ==========================================
# STEP 4: PREDICT ALL TEST IMAGES
# ==========================================

probabilities = model.predict(
    test_generator,
    verbose=1
).ravel()


# Sigmoid score >= 0.5 means PNEUMONIA
predictions = (probabilities >= 0.5).astype(int)

# Actual labels
true_labels = test_generator.classes

print("\n===== SCORES BY ACTUAL CLASS =====")

normal_scores = probabilities[true_labels == 0]
pneumonia_scores = probabilities[true_labels == 1]

print("\nNORMAL images:")
print("Count:", len(normal_scores))
print("Mean score:", float(normal_scores.mean()))
print("Minimum score:", float(normal_scores.min()))
print("Maximum score:", float(normal_scores.max()))

print("\nPNEUMONIA images:")
print("Count:", len(pneumonia_scores))
print("Mean score:", float(pneumonia_scores.mean()))
print("Minimum score:", float(pneumonia_scores.min()))
print("Maximum score:", float(pneumonia_scores.max()))


# ==========================================
# STEP 5: CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


# ==========================================
# STEP 6: CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ==========================================
# STEP 7: PRINT RESULTS
# ==========================================

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================")

print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score:    {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ==========================================
# STEP 8: DISPLAY CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("CNN Test Dataset Confusion Matrix")

plt.tight_layout()

plt.show()


# ==========================================
# STEP 9: SHOW MISCLASSIFIED IMAGES
# ==========================================

# The generator preserves file order because shuffle=False.
file_paths = test_generator.filepaths

misclassified_indices = np.flatnonzero(
    predictions != true_labels
)

print("\nTotal misclassified images:",
      len(misclassified_indices))

if len(misclassified_indices) > 0:

    # Display at most six examples.
    selected_indices = misclassified_indices[:6]

    plt.figure(figsize=(15, 8))

    for plot_index, image_index in enumerate(
        selected_indices,
        start=1
    ):

        image = tf.keras.utils.load_img(
            file_paths[image_index],
            color_mode="grayscale",
            target_size=IMAGE_SIZE
        )

        image = tf.keras.utils.img_to_array(image) / 255.0

        actual = CLASS_NAMES[true_labels[image_index]]

        predicted = CLASS_NAMES[predictions[image_index]]

        score = probabilities[image_index]

        plt.subplot(2, 3, plot_index)

        plt.imshow(image.squeeze(), cmap="gray")

        plt.title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Pneumonia score: {score:.3f}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()

else:
    print("No misclassified images found.")

print("\nEvaluation completed!")