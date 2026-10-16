import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# STEP 1: DETECT PROJECT DIRECTORY
# ============================================================

# main.py is directly inside the project root directory.
BASE_DIR = Path(__file__).resolve().parent

print("=" * 60)
print("CHEST X-RAY IMAGE LOADER")
print("=" * 60)

print(f"\nProject directory:\n{BASE_DIR}")


# ============================================================
# STEP 2: DEFINE THE IMAGE PATH
# ============================================================

IMAGE_NAME = "IM-0115-0001.jpeg"

IMAGE_PATH = (
    BASE_DIR
    / "chest_xray"
    / "train"
    / "NORMAL"
    / IMAGE_NAME
)

print(f"\nImage path:\n{IMAGE_PATH}")
print(f"Image exists: {IMAGE_PATH.exists()}")


# ============================================================
# STEP 3: CHECK WHETHER THE DATASET FOLDER EXISTS
# ============================================================

DATASET_DIR = BASE_DIR / "chest_xray"
NORMAL_DIR = DATASET_DIR / "train" / "NORMAL"
PNEUMONIA_DIR = DATASET_DIR / "train" / "PNEUMONIA"

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder was not found!")
    print(f"Expected location: {DATASET_DIR}")
    print("\nMake sure your project structure looks like this:")
    print("""
    Medical_Image_Analysis/
    ├── main.py
    ├── app.py
    ├── chest_xray/
    │   ├── train/
    │   │   ├── NORMAL/
    │   │   └── PNEUMONIA/
    │   ├── test/
    │   └── val/
    └── models/
    """)
    sys.exit(1)


# ============================================================
# STEP 4: CHECK WHETHER THE IMAGE EXISTS
# ============================================================

if not NORMAL_DIR.exists():
    print("\nERROR: NORMAL training folder was not found!")
    print(f"Expected folder: {NORMAL_DIR}")
    sys.exit(1)

if not IMAGE_PATH.exists():

    print("\nERROR: Image not found!")
    print(f"Expected image: {IMAGE_PATH}")

    # Show some filenames to help identify the correct image.
    print("\nHere are the first 10 images in the NORMAL folder:")

    image_files = [
        file
        for file in NORMAL_DIR.iterdir()
        if file.is_file()
    ]

    if image_files:
        for file in image_files[:10]:
            print(f"  - {file.name}")
    else:
        print("The NORMAL folder is empty.")

    # Search the entire dataset for the requested filename.
    print("\nSearching the dataset for the requested image...")

    matches = list(DATASET_DIR.rglob(IMAGE_NAME))

    if matches:
        print("\nImage found at another location:")
        for match in matches:
            print(f"  - {match}")
        print("\nUpdate IMAGE_PATH to use the correct location.")
    else:
        print(f"\nNo file named '{IMAGE_NAME}' was found in the dataset.")
        print("Check the actual image filename and extension.")

    sys.exit(1)


# ============================================================
# STEP 5: LOAD THE IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    print("\nERROR: OpenCV could not load the image.")
    print("Check whether the image file is valid.")
    sys.exit(1)


# ============================================================
# STEP 6: DISPLAY IMAGE INFORMATION
# ============================================================

height, width = image.shape

print("\nImage loaded successfully!")
print("-" * 60)
print(f"Filename:       {IMAGE_PATH.name}")
print(f"Image shape:    {image.shape}")
print(f"Image height:   {height} pixels")
print(f"Image width:    {width} pixels")
print(f"Image size:     {IMAGE_PATH.stat().st_size / 1024:.2f} KB")
print(f"Image type:     {image.dtype}")
print(f"Minimum pixel:  {image.min()}")
print(f"Maximum pixel:  {image.max()}")
print("-" * 60)


# ============================================================
# STEP 7: DISPLAY THE CHEST X-RAY
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(image, cmap="gray")

plt.title(
    "Chest X-Ray - NORMAL Training Image",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")

plt.axis("off")
plt.tight_layout()

plt.show()


# ============================================================
# STEP 8: FINISH
# ============================================================

print("\nImage processing completed successfully!")