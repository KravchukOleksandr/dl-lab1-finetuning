from __future__ import annotations

import cv2
import numpy as np


def gaussian_weights_1d(
    values: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian weights from scalar values to scalar bin centers.

    Each row is normalized to sum to one.
    """
    dist = values[:, None] - centers[None, :]
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    return weights


def compute_soft_brightness_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    num_bins: int = 6,
    sigma_bins: float = 1.5,
    low_chroma_center: float | None = None,
    low_chroma_sharpness: float | None = None,
    high_chroma_center: float | None = 5.0,
    high_chroma_sharpness: float | None = 0.6,
) -> np.ndarray:
    """
    Compute a soft brightness histogram weighted by chroma selection.

    Pixels are softly distributed over L bins with a Gaussian kernel. The total
    contribution of each pixel is additionally scaled by the universal
    `chroma_weight(...)`, typically configured to keep low-chroma pixels.

    The histogram is normalized by the full working-mask area.
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
    weights *= chroma_weight(
        chroma=chroma,
        low_center=low_chroma_center,
        low_sharpness=low_chroma_sharpness,
        high_center=high_chroma_center,
        high_sharpness=high_chroma_sharpness,
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


def compare_truck_brightness_ratio(
    image1: np.ndarray,
    mask1: np.ndarray,
    src_pts: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    dst_pts: np.ndarray,
    top_fraction: float = 0.45,
    num_bins: int = 6,
    sigma_bins: float = 1.5,
    low_chroma_center: float | None = None,
    low_chroma_sharpness: float | None = None,
    high_chroma_center: float | None = 5.0,
    high_chroma_sharpness: float | None = 0.6,
    min_fraction: float = 0.20,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Full brightness-comparison pipeline.

    The working masks are prepared inside this function using the existing
    `prepare_work_mask(...)`.
    """
    work_mask1 = prepare_work_mask(mask1, src_pts, top_fraction=top_fraction)
    work_mask2 = prepare_work_mask(mask2, dst_pts, top_fraction=top_fraction)

    hist1 = compute_soft_brightness_histogram(
        image=image1,
        mask=work_mask1,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        low_chroma_center=low_chroma_center,
        low_chroma_sharpness=low_chroma_sharpness,
        high_chroma_center=high_chroma_center,
        high_chroma_sharpness=high_chroma_sharpness,
    )

    hist2 = compute_soft_brightness_histogram(
        image=image2,
        mask=work_mask2,
        num_bins=num_bins,
        sigma_bins=sigma_bins,
        low_chroma_center=low_chroma_center,
        low_chroma_sharpness=low_chroma_sharpness,
        high_chroma_center=high_chroma_center,
        high_chroma_sharpness=high_chroma_sharpness,
    )

    ratio_12 = directional_ratio(hist1, hist2, min_fraction=min_fraction)
    ratio_21 = directional_ratio(hist2, hist1, min_fraction=min_fraction)
    ratio_min = min(ratio_12, ratio_21)

    return ratio_12, ratio_21, ratio_min, hist1, hist2


TOP_FRACTION = 0.45

NUM_BINS = 6
SIGMA_BINS = 1.5

LOW_CHROMA_CENTER = None
LOW_CHROMA_SHARPNESS = None

HIGH_CHROMA_CENTER = 5.0
HIGH_CHROMA_SHARPNESS = 0.6

MIN_FRACTION = 0.20

ratio_12, ratio_21, ratio_min, hist1, hist2 = compare_truck_brightness_ratio(
    image1=img1,
    mask1=mask1,
    src_pts=src_pts,
    image2=img2,
    mask2=mask2,
    dst_pts=dst_pts,
    top_fraction=TOP_FRACTION,
    num_bins=NUM_BINS,
    sigma_bins=SIGMA_BINS,
    low_chroma_center=LOW_CHROMA_CENTER,
    low_chroma_sharpness=LOW_CHROMA_SHARPNESS,
    high_chroma_center=HIGH_CHROMA_CENTER,
    high_chroma_sharpness=HIGH_CHROMA_SHARPNESS,
    min_fraction=MIN_FRACTION,
)

print("brightness ratio 1->2:", ratio_12)
print("brightness ratio 2->1:", ratio_21)
print("brightness ratio final:", ratio_min)
print("brightness hist1:", hist1)
print("brightness hist2:", hist2)