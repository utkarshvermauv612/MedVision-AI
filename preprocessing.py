# ==========================================
# MEDICAL IMAGE PREPROCESSING
# Chest X-Ray Pneumonia Detection
# ==========================================

# Import required libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================
# STEP 1: DEFINE DATASET PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "chest_xray")

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")


# ==========================================
# STEP 2: DEFINE IMAGE SETTINGS
# ==========================================

IMAGE_SIZE = (256, 256)

BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# STEP 3: DATA AUGMENTATION FOR TRAINING
# ==========================================

train_datagen = ImageDataGenerator(

    # Normalize pixels between 0 and 1
    rescale=1.0 / 255,

    # Random rotation
    rotation_range=15,

    # Random zoom
    zoom_range=0.15,

    # Random horizontal flipping
    horizontal_flip=True,

    # Fill missing pixels
    fill_mode="nearest",

    validation_split=0.2
)


# ==========================================
# STEP 4: VALIDATION AND TEST PREPROCESSING
# ==========================================

# No augmentation for validation/test data

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ==========================================
# STEP 5: LOAD TRAINING DATA
# ==========================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=CLASS_NAMES,

    shuffle=True
)


# ==========================================
# STEP 6: LOAD VALIDATION DATA
# ==========================================

val_generator = val_test_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMAGE_SIZE,

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=CLASS_NAMES,

    shuffle=False
)


# ==========================================
# STEP 7: LOAD TESTING DATA
# ==========================================

test_generator = val_test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=CLASS_NAMES,

    shuffle=False
)


# ==========================================
# STEP 8: DISPLAY DATASET INFORMATION
# ==========================================

print("\n===================================")
print("       DATASET INFORMATION")
print("===================================")

print("Training Images:", train_generator.samples)

print("Validation Images:", val_generator.samples)

print("Testing Images:", test_generator.samples)

print("\nClass Labels:")

print(train_generator.class_indices)


# ==========================================
# STEP 9: LOAD SAMPLE ORIGINAL IMAGE
# ==========================================

sample_folder = os.path.join(TRAIN_DIR, "NORMAL")

image_files = [
    f for f in os.listdir(sample_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise FileNotFoundError(
        "No images found in train/NORMAL folder."
    )

sample_image_path = os.path.join(
    sample_folder,
    image_files[0]
)

# Load original image in grayscale
original_image = cv2.imread(
    sample_image_path,
    cv2.IMREAD_GRAYSCALE
)

if original_image is None:
    raise ValueError("Unable to load sample image.")

# Resize to 256x256
resized_image = cv2.resize(
    original_image,
    IMAGE_SIZE
)

# Normalize image
normalized_image = resized_image.astype("float32") / 255.0


# ==========================================
# STEP 10: GENERATE AUGMENTED IMAGES
# ==========================================

# Reshape image for augmentation
image_array = normalized_image.reshape(
    1, 256, 256, 1
)

# Separate generator for visualization
# Input is already normalized, so no rescale here
augmentation_generator = ImageDataGenerator(

    rotation_range=15,

    zoom_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest"
)

augmented_images = []

augmentation_iterator = augmentation_generator.flow(
    image_array,
    batch_size=1,
    shuffle=False,
    seed=42
)

for i in range(5):

    batch = next(augmentation_iterator)

    augmented_images.append(batch[0])


# ==========================================
# STEP 11: DISPLAY PREPROCESSED IMAGES
# ==========================================

plt.figure(figsize=(15, 8))

# Original Image
plt.subplot(2, 4, 1)

plt.imshow(original_image, cmap="gray")

plt.title("Original X-Ray")

plt.axis("off")


# Resized Image
plt.subplot(2, 4, 2)

plt.imshow(resized_image, cmap="gray")

plt.title("Resized 256x256")

plt.axis("off")


# Normalized Image
plt.subplot(2, 4, 3)

plt.imshow(normalized_image, cmap="gray", vmin=0, vmax=1)

plt.title("Normalized Image")

plt.axis("off")


# Augmented Images
for i in range(5):

    plt.subplot(2, 4, i + 4)

    plt.imshow(
        augmented_images[i].squeeze(),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(f"Augmented {i + 1}")

    plt.axis("off")


plt.tight_layout()

plt.show()


# ==========================================
# STEP 12: CHECK IMAGE BATCH SHAPE
# ==========================================

images, labels = next(train_generator)

print("\n===================================")
print("      PREPROCESSING RESULTS")
print("===================================")

print("Batch Image Shape:", images.shape)

print("Batch Label Shape:", labels.shape)

print("Minimum Pixel Value:", images.min())

print("Maximum Pixel Value:", images.max())

print("\nPreprocessing Completed Successfully!")