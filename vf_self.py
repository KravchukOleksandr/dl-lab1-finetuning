from __future__ import annotations

import cv2
import numpy as np


def apply_deterministic_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply one fixed deterministic augmentation pipeline to a BGR image.

    The function is intended for feature stability checks, not for training.
    It always applies the same sequence of light augmentations and returns
    one augmented BGR uint8 image.

    Args:
        image:
            Input image as a BGR uint8 numpy array.

    Returns:
        Augmented image as a BGR uint8 numpy array.
    """
    h, w = image.shape[:2]

    blur_kernel = (3, 3)
    blur_sigma = 0.8

    brightness_factor = 0.95
    contrast_factor = 0.95

    jpeg_quality = 85

    resize_scale = 0.92

    affine_angle_deg = 1.0
    affine_scale = 0.99
    affine_shift_x = 0.01 * w
    affine_shift_y = 0.01 * h

    augmented = cv2.GaussianBlur(image, blur_kernel, blur_sigma)

    augmented = np.clip(augmented.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    augmented = np.clip((augmented.astype(np.float32) - 127.5) * contrast_factor + 127.5, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(
        ".jpg",
        augmented,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        return augmented

    augmented = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if augmented is None:
        return image.copy()

    small_w = max(1, int(round(w * resize_scale)))
    small_h = max(1, int(round(h * resize_scale)))

    augmented = cv2.resize(augmented, (small_w, small_h), interpolation=cv2.INTER_AREA)
    augmented = cv2.resize(augmented, (w, h), interpolation=cv2.INTER_LINEAR)

    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), affine_angle_deg, affine_scale)
    matrix[0, 2] += affine_shift_x
    matrix[1, 2] += affine_shift_y

    augmented = cv2.warpAffine(
        augmented,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return augmented