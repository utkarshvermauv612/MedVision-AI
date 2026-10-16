
"""
MedVision AI - Image Quality Validator

Checks whether an uploaded image:
1. Is a valid JPG, JPEG, or PNG image.
2. Has sufficient dimensions.
3. Is not excessively dark or bright.
4. Has sufficient contrast.
5. Is not excessively blurry.

IMPORTANT:
These are basic image-quality checks, not a medical validation system.
They cannot determine whether an image is a chest X-ray.
Quality thresholds must be tested and calibrated against your dataset.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# 1. CONFIGURATION
# ============================================================

MIN_IMAGE_WIDTH = 150
MIN_IMAGE_HEIGHT = 150

# Initial heuristic thresholds; calibrate against real X-rays.
MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 245
MIN_CONTRAST = 10

# Laplacian variance below this value may indicate blur.
# This threshold is not clinically validated.
MIN_SHARPNESS = 8

SUPPORTED_FORMATS = {"JPEG", "PNG"}


# ============================================================
# 2. DECODE THE UPLOADED IMAGE
# ============================================================

def decode_uploaded_image(file_bytes):
    """
    Decode an uploaded JPG or PNG image.

    Returns:
        PIL.Image.Image: RGB image.

    Raises:
        ValueError: If the image is invalid or unsupported.
    """

    if not file_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(file_bytes)) as source:

            if source.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Unsupported image format. "
                    "Please upload a JPG, JPEG, or PNG image."
                )

            # Correct image orientation using EXIF metadata.
            image = ImageOps.exif_transpose(source)

            # Decode the image before closing the file.
            image.load()

            image = image.convert("RGB")

    except UnidentifiedImageError as exc:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from exc

    except OSError as exc:
        raise ValueError(
            "The uploaded image could not be decoded."
        ) from exc

    return image


# ============================================================
# 3. CALCULATE IMAGE QUALITY METRICS
# ============================================================

def calculate_image_quality(image):
    """
    Calculate basic image-quality measurements.

    Returns:
        dict: Image dimensions, brightness, contrast and sharpness.
    """

    grayscale = image.convert("L")

    grayscale_array = np.asarray(
        grayscale,
        dtype=np.float32,
    )

    height, width = grayscale_array.shape

    brightness = float(np.mean(grayscale_array))
    contrast = float(np.std(grayscale_array))

    # Calculate the variance of the Laplacian without requiring
    # OpenCV. This is a rough indicator of image sharpness.
    padded = np.pad(
        grayscale_array,
        pad_width=1,
        mode="edge",
    )

    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )

    sharpness = float(np.var(laplacian))

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
    }


# ============================================================
# 4. VALIDATE IMAGE QUALITY
# ============================================================

def validate_image_quality(image):
    """
    Check basic image-quality requirements.

    Returns:
        dict: Validation status, quality metrics and messages.
    """

    metrics = calculate_image_quality(image)
    errors = []

    # Check image dimensions.
    if (
        metrics["width"] < MIN_IMAGE_WIDTH
        or metrics["height"] < MIN_IMAGE_HEIGHT
    ):
        errors.append(
            "Image resolution is too low. "
            f"Minimum dimensions are "
            f"{MIN_IMAGE_WIDTH} x {MIN_IMAGE_HEIGHT} pixels."
        )

    # Check whether the image is excessively dark.
    if metrics["brightness"] < MIN_BRIGHTNESS:
        errors.append(
            "The image is too dark. "
            "Please upload a properly exposed image."
        )

    # Check whether the image is excessively bright.
    if metrics["brightness"] > MAX_BRIGHTNESS:
        errors.append(
            "The image is too bright. "
            "Please upload a properly exposed image."
        )

    # Check whether the image has extremely low contrast.
    if metrics["contrast"] < MIN_CONTRAST:
        errors.append(
            "The image has insufficient contrast. "
            "Please upload a clearer image."
        )

    # Check whether the image may be excessively blurry.
    if metrics["sharpness"] < MIN_SHARPNESS:
        errors.append(
            "The image may be too blurry. "
            "Please upload a sharper image."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "quality": metrics,
    }


# ============================================================
# 5. COMPLETE VALIDATION FUNCTION
# ============================================================

def validate_uploaded_image(file_bytes):
    """
    Decode and validate an uploaded image.

    Returns:
        dict:
            valid: Whether basic quality checks passed.
            errors: List of validation errors.
            quality: Image-quality measurements.
            image: Decoded RGB image, or None if decoding failed.
    """

    try:
        image = decode_uploaded_image(file_bytes)

    except ValueError as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "quality": None,
            "image": None,
        }

    result = validate_image_quality(image)

    result["image"] = image

    return result