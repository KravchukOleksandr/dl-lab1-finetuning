from __future__ import annotations

import cv2
import numpy as np


def crop_mask_by_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Keep only the tight keypoint area inside the mask.
    """
    x = points[:, 0]
    y = points[:, 1]

    x1 = max(0, int(np.floor(x.min())))
    x2 = min(mask.shape[1], int(np.ceil(x.max())))
    y1 = max(0, int(np.floor(y.min())))
    y2 = min(mask.shape[0], int(np.ceil(y.max())))

    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def cut_top_fraction(mask: np.ndarray, top_fraction: float = 0.45) -> np.ndarray:
    """
    Remove the top fraction of the mask.
    """
    h = mask.shape[0]
    out = mask.copy()
    out[: int(round(h * top_fraction)), :] = 0
    return out


def prepare_color_mask(
    mask: np.ndarray,
    points: np.ndarray,
    top_fraction: float = 0.45,
) -> np.ndarray:
    """
    Crop the mask by keypoints and remove the top part.
    """
    return cut_top_fraction(crop_mask_by_points(mask, points), top_fraction=top_fraction)


def compute_color_window_fractions(
    image: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 16,
) -> np.ndarray:
    """
    Compute circular 3-bin window fractions over color angle in Lab a,b space.

    Only non-neutral hue-like color angle is used here. L is intentionally ignored.

    Returns:
        Array of shape (num_bins,) with window fractions.
    """
    valid = mask > 0
    total = int(valid.sum())
    if total == 0:
        return np.zeros(num_bins, dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1][valid] - 128.0
    b = lab[..., 2][valid] - 128.0

    angles = np.arctan2(b, a)
    angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)

    bins = np.floor(angles / (2.0 * np.pi) * num_bins).astype(np.int32)
    bins = np.clip(bins, 0, num_bins - 1)

    hist = np.bincount(bins, minlength=num_bins).astype(np.float32)
    hist /= hist.sum() + 1e-6

    windows = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        windows[i] = hist[(i - 1) % num_bins] + hist[i] + hist[(i + 1) % num_bins]

    return windows


def directional_color_ratio(
    windows1: np.ndarray,
    windows2: np.ndarray,
    min_fraction: float = 0.20,
) -> float:
    """
    Compute the worst retained ratio from truck1 to truck2.

    For every window that is significant on truck1, compute:
        windows2[i] / windows1[i]

    Return the minimum ratio across all significant windows.

    If truck1 has no significant windows, return 1.0.
    """
    valid = windows1 >= min_fraction
    if not np.any(valid):
        return 1.0

    ratios = windows2[valid] / (windows1[valid] + 1e-6)
    return float(ratios.min())


def compare_truck_color_ratio(
    image1: np.ndarray,
    mask1: np.ndarray,
    src_pts: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    dst_pts: np.ndarray,
    top_fraction: float = 0.45,
    num_bins: int = 16,
    min_fraction: float = 0.20,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compare truck colors and return directional and symmetric ratios.

    Returns:
        ratio_12:
            Worst retained ratio from truck1 to truck2.
        ratio_21:
            Worst retained ratio from truck2 to truck1.
        ratio_min:
            min(ratio_12, ratio_21)
        windows1:
            3-bin circular window fractions for truck1.
        windows2:
            3-bin circular window fractions for truck2.
    """
    work_mask1 = prepare_color_mask(mask1, src_pts, top_fraction=top_fraction)
    work_mask2 = prepare_color_mask(mask2, dst_pts, top_fraction=top_fraction)

    windows1 = compute_color_window_fractions(image1, work_mask1, num_bins=num_bins)
    windows2 = compute_color_window_fractions(image2, work_mask2, num_bins=num_bins)

    ratio_12 = directional_color_ratio(windows1, windows2, min_fraction=min_fraction)
    ratio_21 = directional_color_ratio(windows2, windows1, min_fraction=min_fraction)
    ratio_min = min(ratio_12, ratio_21)

    return ratio_12, ratio_21, ratio_min, windows1, windows2


ratio_12, ratio_21, ratio_min, windows1, windows2 = compare_truck_color_ratio(
    image1=img1,
    mask1=mask1,
    src_pts=src_pts,
    image2=img2,
    mask2=mask2,
    dst_pts=dst_pts,
    top_fraction=0.45,
    num_bins=16,
    min_fraction=0.20,
)

print("ratio 1->2:", ratio_12)
print("ratio 2->1:", ratio_21)
print("final ratio:", ratio_min)