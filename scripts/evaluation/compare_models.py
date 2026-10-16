import os
import numpy as np
import tensorflow as tf

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATHS = {
    "best_model": r"A:\Project\Medical_Image_Analysis\models\best_model.keras",
    "final_model": r"A:\Project\Medical_Image_Analysis\models\final_model.keras"
}

TEST_DIR = r"A:\Project\Medical_Image_Analysis\chest_xray\test"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# LOAD TEST DATA
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass mapping:", test_ds.class_names)


# ==========================================
# EVALUATE BOTH MODELS
# ==========================================

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 55)
    print("MODEL:", model_name)
    print("=" * 55)

    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        continue

    model = tf.keras.models.load_model(model_path)

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    scores = []
    labels = []

    for images, batch_labels in test_ds:

        predictions = model(images, training=False)

        scores.extend(
            np.asarray(predictions).reshape(-1)
        )

        labels.extend(
            batch_labels.numpy().reshape(-1)
        )

    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    print("\nOverall predictions:")
    print("Minimum:", np.min(scores))
    print("Maximum:", np.max(scores))
    print("Mean:", np.mean(scores))
    print("Standard deviation:", np.std(scores))

    for class_id, class_name in enumerate(CLASS_NAMES):

        class_scores = scores[labels == class_id]

        print(f"\n{class_name}:")
        print("Number of images:", len(class_scores))

        if len(class_scores):
            print("Mean score:", np.mean(class_scores))
            print("Minimum score:", np.min(class_scores))
            print("Maximum score:", np.max(class_scores))
            print("Standard deviation:", np.std(class_scores))

    print("\nNumber of scores below 0.5:", np.sum(scores < 0.5))
    print("Number of scores at or above 0.5:", np.sum(scores >= 0.5))

print("\nModel comparison completed.")