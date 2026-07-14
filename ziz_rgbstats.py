from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def centered_inner_square(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    height, width = image.shape[:2]
    side = min(height, width)

    x1 = (width - side) // 2
    y1 = (height - side) // 2

    return image[y1:y1 + side, x1:x1 + side]


def resize_square(
    square: np.ndarray,
    output_size: int = 64,
) -> np.ndarray:
    side = square.shape[0]

    interpolation = (
        cv2.INTER_AREA
        if side > output_size
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        square,
        (output_size, output_size),
        interpolation=interpolation,
    )


def calculate_train_mean_std(
    train_directory: str | Path,
    output_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    train_directory = Path(train_directory)

    paths = [
        path
        for path in train_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not paths:
        raise ValueError(f"No images found in {train_directory}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")

        square = centered_inner_square(image_bgr)
        square = resize_square(square, output_size)

        image_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float64) / 255.0

        channel_sum += image.sum(axis=(0, 1))
        channel_squared_sum += np.square(image).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


mean, std = calculate_train_mean_std(
    "ziz-crops-202607-2/train",
    output_size=64,
)

print("mean:", mean.tolist())
print("std:", std.tolist())