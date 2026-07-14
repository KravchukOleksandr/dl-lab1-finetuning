"""
Preprocessing and dataset code for helmet classification.

Class mapping used by the whole project:
    0 = helmet     (каска есть)
    1 = no_helmet  (каски нет; positive class for metrics)

Expected dataset structure:
    DATASET_ROOT/
        train/
            helmet/
            no-helmet/   # also accepts no_helmet or nohelmet
        val/
            helmet/
            no-helmet/

Images on disk are expected to be the already-expanded x1.25 crops.
No anisotropic resize is used: first a square is cropped, then that square is
isotropically resized to IMAGE_SIZE x IMAGE_SIZE.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Class mapping. Keep this unchanged across training and production.
# -----------------------------------------------------------------------------
CLASS_NAMES: dict[int, str] = {
    0: "helmet",
    1: "no_helmet",
}

CLASS_DIRECTORY_ALIASES: dict[int, tuple[str, ...]] = {
    0: ("helmet",),
    1: ("no-helmet", "no_helmet", "nohelmet"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Train-set statistics supplied by the user. Values are in RGB order and were
# measured after: central square -> isotropic resize -> division by 255.
DEFAULT_MEAN = (0.5226514935493469, 0.4618634879589081, 0.4670388400554657)
DEFAULT_STD = (0.2495836615562439, 0.2355685979127884, 0.23572568595409393)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def centered_inner_square(image: np.ndarray) -> np.ndarray:
    """Return the largest centered square fully contained in the image."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    height, width = image.shape[:2]
    side = min(height, width)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return image[y1 : y1 + side, x1 : x1 + side]


def _coverage_of_original_bbox(
    crop_box: tuple[float, float, float, float],
    original_box: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """
    Calculate x/y coverage of the approximated original bbox and top loss.

    The stored image is an x1.25 expansion around the original bbox, so the
    original bbox approximately occupies the central 80% on both axes.
    """
    cx1, cy1, cx2, cy2 = crop_box
    ox1, oy1, ox2, oy2 = original_box

    inter_w = max(0.0, min(cx2, ox2) - max(cx1, ox1))
    inter_h = max(0.0, min(cy2, oy2) - max(cy1, oy1))

    original_w = max(ox2 - ox1, 1e-6)
    original_h = max(oy2 - oy1, 1e-6)

    coverage_x = inter_w / original_w
    coverage_y = inter_h / original_h
    top_loss = max(0.0, cy1 - oy1) / original_h
    return coverage_x, coverage_y, top_loss


def random_safe_inner_square(
    image: np.ndarray,
    min_side_scale: float = 0.90,
    max_side_scale: float = 1.00,
    center_jitter: float = 0.03,
    min_original_coverage: float = 0.90,
    max_top_loss: float = 0.05,
    attempts: int = 20,
) -> np.ndarray:
    """
    Crop a mildly randomized square from an already x1.25-expanded crop.

    - square side is 90-100% of the largest possible inner square;
    - center moves by at most +/-3% of the maximum square side;
    - at least 90% of the approximated original bbox is preserved on each axis;
    - no more than 5% of its top is removed.

    If no random candidate satisfies the constraints, the deterministic largest
    centered square is returned.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if not (0.0 < min_side_scale <= max_side_scale <= 1.0):
        raise ValueError("Expected 0 < min_side_scale <= max_side_scale <= 1")

    height, width = image.shape[:2]
    max_side = min(height, width)

    # x1.25 expansion means the original bbox is approximately the central 80%.
    original_box = (
        0.10 * width,
        0.10 * height,
        0.90 * width,
        0.90 * height,
    )

    shift_limit = center_jitter * max_side

    for _ in range(attempts):
        side = int(round(max_side * random.uniform(min_side_scale, max_side_scale)))
        side = max(1, min(side, max_side))

        center_x = 0.5 * width + random.uniform(-shift_limit, shift_limit)
        center_y = 0.5 * height + random.uniform(-shift_limit, shift_limit)

        x1 = int(round(center_x - 0.5 * side))
        y1 = int(round(center_y - 0.5 * side))

        # Keep the randomized square fully inside the saved crop.
        x1 = min(max(x1, 0), width - side)
        y1 = min(max(y1, 0), height - side)
        x2 = x1 + side
        y2 = y1 + side

        coverage_x, coverage_y, top_loss = _coverage_of_original_bbox(
            (x1, y1, x2, y2),
            original_box,
        )

        if (
            coverage_x >= min_original_coverage
            and coverage_y >= min_original_coverage
            and top_loss <= max_top_loss
        ):
            return image[y1:y2, x1:x2]

    return centered_inner_square(image)


def isotropic_resize_square(square: np.ndarray, output_size: int = 64) -> np.ndarray:
    """Resize a square to output_size x output_size without shape distortion."""
    if square is None or square.size == 0:
        raise ValueError("Empty square")
    if square.shape[0] != square.shape[1]:
        raise ValueError(f"Expected a square, got shape={square.shape}")

    side = square.shape[0]
    interpolation = cv2.INTER_AREA if side > output_size else cv2.INTER_LINEAR
    return cv2.resize(square, (output_size, output_size), interpolation=interpolation)


# -----------------------------------------------------------------------------
# Mild photometric augmentation
# -----------------------------------------------------------------------------
def _adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    result = image.astype(np.float32) * factor
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    image_f = image.astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)
    result = mean + factor * (image_f - mean)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    result = np.power(normalized, gamma) * 255.0
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def mild_photometric_augmentation(
    image: np.ndarray,
    probability: float = 0.50,
    brightness_limit: float = 0.10,
    contrast_limit: float = 0.10,
    gamma_range: tuple[float, float] = (0.90, 1.10),
) -> np.ndarray:
    """
    With the given probability, apply exactly one mild operation.

    Applying only one operation avoids compounding brightness, contrast and
    gamma changes on already difficult low-quality images.
    """
    if random.random() >= probability:
        return image

    operation = random.choice(("brightness", "contrast", "gamma"))

    if operation == "brightness":
        factor = random.uniform(1.0 - brightness_limit, 1.0 + brightness_limit)
        return _adjust_brightness(image, factor)

    if operation == "contrast":
        factor = random.uniform(1.0 - contrast_limit, 1.0 + contrast_limit)
        return _adjust_contrast(image, factor)

    gamma = random.uniform(*gamma_range)
    return _adjust_gamma(image, gamma)


# -----------------------------------------------------------------------------
# Complete preprocessing callable
# -----------------------------------------------------------------------------
class HelmetPreprocessor:
    """Convert a stored BGR crop into a normalized RGB tensor."""

    def __init__(
        self,
        training: bool,
        output_size: int = 64,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        square_scale: tuple[float, float] = (0.90, 1.00),
        center_jitter: float = 0.03,
        horizontal_flip_probability: float = 0.50,
        photometric_probability: float = 0.50,
        brightness_limit: float = 0.10,
        contrast_limit: float = 0.10,
        gamma_range: tuple[float, float] = (0.90, 1.10),
    ) -> None:
        self.training = training
        self.output_size = output_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        if np.any(self.std <= 0):
            raise ValueError("All std values must be positive")

        self.square_scale = square_scale
        self.center_jitter = center_jitter
        self.horizontal_flip_probability = horizontal_flip_probability
        self.photometric_probability = photometric_probability
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.gamma_range = gamma_range

    def __call__(self, image_bgr: np.ndarray) -> torch.Tensor:
        if self.training:
            square = random_safe_inner_square(
                image_bgr,
                min_side_scale=self.square_scale[0],
                max_side_scale=self.square_scale[1],
                center_jitter=self.center_jitter,
            )

            if random.random() < self.horizontal_flip_probability:
                square = cv2.flip(square, 1)
        else:
            square = centered_inner_square(image_bgr)

        square = isotropic_resize_square(square, self.output_size)

        if self.training:
            square = mild_photometric_augmentation(
                square,
                probability=self.photometric_probability,
                brightness_limit=self.brightness_limit,
                contrast_limit=self.contrast_limit,
                gamma_range=self.gamma_range,
            )

        image_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HelmetFolderDataset(Dataset):
    """Binary dataset with an explicit, stable class mapping."""

    def __init__(
        self,
        split_directory: str | Path,
        preprocessor: HelmetPreprocessor,
    ) -> None:
        self.split_directory = Path(split_directory)
        self.preprocessor = preprocessor
        self.samples: list[tuple[Path, int]] = []

        if not self.split_directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.split_directory}")

        for label, aliases in CLASS_DIRECTORY_ALIASES.items():
            found_directory = False
            for directory_name in aliases:
                class_directory = self.split_directory / directory_name
                if not class_directory.exists():
                    continue

                found_directory = True
                for path in sorted(class_directory.rglob("*")):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((path, label))

            if not found_directory:
                expected = ", ".join(str(self.split_directory / name) for name in aliases)
                raise FileNotFoundError(
                    f"No directory found for class {label} ({CLASS_NAMES[label]}). "
                    f"Expected one of: {expected}"
                )

        if not self.samples:
            raise RuntimeError(f"No images found in {self.split_directory}")

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path, label = self.samples[index]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        image = self.preprocessor(image_bgr)
        target = torch.tensor(float(label), dtype=torch.float32)
        return image, target, str(path)

    def class_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0}
        for target in self.targets:
            counts[target] += 1
        return counts


# -----------------------------------------------------------------------------
# Optional utility to recompute train mean/std
# -----------------------------------------------------------------------------
def _iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def calculate_train_mean_std(
    train_directory: str | Path,
    output_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RGB mean/std over every train image using deterministic preprocessing.

    No class balancing and no augmentation are applied. This estimates the
    natural train input distribution.
    """
    train_directory = Path(train_directory)
    paths = list(_iter_image_paths(train_directory))
    if not paths:
        raise ValueError(f"No images found in {train_directory}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        square = centered_inner_square(image_bgr)
        square = isotropic_resize_square(square, output_size)
        image_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float64) / 255.0

        channel_sum += image.sum(axis=(0, 1))
        channel_squared_sum += np.square(image).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)
