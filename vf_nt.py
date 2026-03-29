from __future__ import annotations

import cv2
import numpy as np


def crop_mask_by_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Keep only the tight keypoint rectangle inside the mask.
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
    Remove the top part of the mask.
    """
    h = mask.shape[0]
    out = mask.copy()
    out[: int(round(h * top_fraction)), :] = 0
    return out


def prepare_work_mask(
    mask: np.ndarray,
    points: np.ndarray,
    top_fraction: float = 0.45,
) -> np.ndarray:
    """
    Crop the mask by keypoints and remove the top part.
    """
    return cut_top_fraction(
        crop_mask_by_points(mask, points),
        top_fraction=top_fraction,
    )


def chroma_weight(
    chroma: np.ndarray,
    center: float = 15.0,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Smooth color weight as a function of chroma.

    The curve is near zero for weak colors and quickly approaches one for
    sufficiently saturated colors.
    """
    return 0.5 * (1.0 + np.tanh((chroma - center) / sharpness))


def neutral_weight(
    chroma: np.ndarray,
    center: float = 15.0,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Smooth neutral weight as a function of chroma.

    This is the complement of `chroma_weight`.
    """
    return 1.0 - chroma_weight(chroma, center=center, sharpness=sharpness)


def circular_angle_distance(theta: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Absolute circular distance between angles and circular bin centers.
    """
    delta = theta[:, None] - centers[None, :]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(delta)


def gaussian_weights_1d(
    values: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian weights from scalar values to scalar bin centers.
    """
    dist = values[:, None] - centers[None, :]
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    return weights


def compute_soft_color_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 16,
    sigma_bins: float = 0.9,
    chroma_center: float = 15.0,
    chroma_sharpness: float = 3.0,
) -> np.ndarray:
    """
    Compute a soft circular hue histogram in Lab a,b space.

    Each pixel inside the mask contributes to all hue bins through a circular
    Gaussian kernel. The contribution magnitude is scaled by `chroma_weight`.

    The histogram is normalized by the full mask area, so the total mass
    reflects how much colored content is present in the working mask.
    """
    valid = mask > 0
    total_area = int(valid.sum())
    if total_area == 0:
        return np.zeros(num_bins, dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    a = lab[..., 1][valid] - 128.0
    b = lab[..., 2][valid] - 128.0

    chroma = np.sqrt(a * a + b * b)
    theta = np.arctan2(b, a)
    theta = (theta + 2.0 * np.pi) % (2.0 * np.pi)

    centers = 2.0 * np.pi * np.arange(num_bins, dtype=np.float32) / num_bins
    sigma_rad = sigma_bins * (2.0 * np.pi / num_bins)

    dist = circular_angle_distance(theta, centers)
    weights = np.exp(-0.5 * (dist / sigma_rad) ** 2)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12

    weights *= chroma_weight(
        chroma,
        center=chroma_center,
        sharpness=chroma_sharpness,
    )[:, None]

    hist = weights.sum(axis=0).astype(np.float32)
    hist /= float(total_area)

    return hist


def compute_soft_neutral_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 6,
    sigma_bins: float = 1.5,
    chroma_center: float = 15.0,
    chroma_sharpness: float = 3.0,
) -> np.ndarray:
    """
    Compute a soft lightness histogram for neutral content.

    Each pixel inside the mask contributes to all L bins through a Gaussian
    kernel along the L axis. The contribution magnitude is scaled by
    `neutral_weight`.

    The histogram is normalized by the full mask area, so the total mass
    reflects how much neutral content is present in the working mask.
    """
    valid = mask > 0
    total_area = int(valid.sum())
    if total_area == 0:
        return np.zeros(num_bins, dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[..., 0][valid]
    a = lab[..., 1][valid] - 128.0
    b = lab[..., 2][valid] - 128.0

    chroma = np.sqrt(a * a + b * b)

    bin_step = 256.0 / num_bins
    centers = (np.arange(num_bins, dtype=np.float32) + 0.5) * bin_step
    sigma_L = sigma_bins * bin_step

    weights = gaussian_weights_1d(L, centers, sigma=sigma_L)
    weights *= neutral_weight(
        chroma,
        center=chroma_center,
        sharpness=chroma_sharpness,
    )[:, None]

    hist = weights.sum(axis=0).astype(np.float32)
    hist /= float(total_area)

    return hist


def directional_ratio(
    hist1: np.ndarray,
    hist2: np.ndarray,
    min_fraction: float = 0.20,
) -> float:
    """
    Compute the worst retained ratio from hist1 to hist2.

    Only bins with mass >= min_fraction in hist1 are checked. For each such bin:
        ratio_i = hist2[i] / hist1[i]

    The returned value is the minimum ratio over all significant bins.

    If hist1 has no significant bins, the function returns 1.0.
    """
    significant = hist1 >= min_fraction
    if not np.any(significant):
        return 1.0

    ratios = hist2[significant] / (hist1[significant] + 1e-12)
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
    sigma_bins: float = 0.9,
    chroma_center: float = 15.0,
    chroma_sharpness: float = 3.0,
    min_fraction: float = 0.20,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Full saturated-color comparison pipeline.

    Returns:
        ratio_12:
            Worst retained ratio from truck1 to truck2.
        ratio_21:
            Worst retained ratio from truck2 to truck1.
        ratio_min:
            min(ratio_12, ratio_21)
        hist1:
            Soft color histogram for truck1.
        hist2:
            Soft color histogram for truck2.
    """
    work_mask1 = prepare_work_mask(mask1, src_pts, top_fraction=top_fraction)
    work_mask2 = prepare_work_mask(mask2, dst_pts, top_fraction=top_fraction)

    hist1 = compute_soft_color_histogram(
        image=image1,
        mask=work_mask1,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        chroma_center=chroma_center,
        chroma_sharpness=chroma_sharpness,
    )
    hist2 = compute_soft_color_histogram(
        image=image2,
        mask=work_mask2,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        chroma_center=chroma_center,
        chroma_sharpness=chroma_sharpness,
    )

    ratio_12 = directional_ratio(hist1, hist2, min_fraction=min_fraction)
    ratio_21 = directional_ratio(hist2, hist1, min_fraction=min_fraction)
    ratio_min = min(ratio_12, ratio_21)

    return ratio_12, ratio_21, ratio_min, hist1, hist2


def compare_truck_neutral_ratio(
    image1: np.ndarray,
    mask1: np.ndarray,
    src_pts: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    dst_pts: np.ndarray,
    top_fraction: float = 0.45,
    num_bins: int = 6,
    sigma_bins: float = 1.5,
    chroma_center: float = 15.0,
    chroma_sharpness: float = 3.0,
    min_fraction: float = 0.20,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Full neutral-tone comparison pipeline.

    Returns:
        ratio_12:
            Worst retained ratio from truck1 to truck2.
        ratio_21:
            Worst retained ratio from truck2 to truck1.
        ratio_min:
            min(ratio_12, ratio_21)
        hist1:
            Soft neutral histogram for truck1.
        hist2:
            Soft neutral histogram for truck2.
    """
    work_mask1 = prepare_work_mask(mask1, src_pts, top_fraction=top_fraction)
    work_mask2 = prepare_work_mask(mask2, dst_pts, top_fraction=top_fraction)

    hist1 = compute_soft_neutral_histogram(
        image=image1,
        mask=work_mask1,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        chroma_center=chroma_center,
        chroma_sharpness=chroma_sharpness,
    )
    hist2 = compute_soft_neutral_histogram(
        image=image2,
        mask=work_mask2,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        chroma_center=chroma_center,
        chroma_sharpness=chroma_sharpness,
    )

    ratio_12 = directional_ratio(hist1, hist2, min_fraction=min_fraction)
    ratio_21 = directional_ratio(hist2, hist1, min_fraction=min_fraction)
    ratio_min = min(ratio_12, ratio_21)

    return ratio_12, ratio_21, ratio_min, hist1, hist2




color_ratio_12, color_ratio_21, color_ratio, color_hist1, color_hist2 = compare_truck_color_ratio(
    image1=img1,
    mask1=mask1,
    src_pts=src_pts,
    image2=img2,
    mask2=mask2,
    dst_pts=dst_pts,
)

neutral_ratio_12, neutral_ratio_21, neutral_ratio, neutral_hist1, neutral_hist2 = compare_truck_neutral_ratio(
    image1=img1,
    mask1=mask1,
    src_pts=src_pts,
    image2=img2,
    mask2=mask2,
    dst_pts=dst_pts,
)

print("color ratio:", color_ratio)
print("neutral ratio:", neutral_ratio)