from __future__ import annotations

import cv2
import numpy as np


def crop_mask_by_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Keep only the tight keypoint rectangle inside the mask.

    Args:
        mask:
            Binary mask of shape (H, W).
        points:
            Array of shape (N, 2) with point coordinates in (x, y) format.

    Returns:
        Mask of the same shape, zeroed outside the tight point bbox.
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

    Args:
        mask:
            Binary mask of shape (H, W).
        top_fraction:
            Fraction of the height to remove from the top.

    Returns:
        Mask of the same shape with the top part zeroed out.
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
    Prepare the working mask for color analysis.

    The mask is first cropped by keypoints, then the top part is removed.
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
    Soft chroma weight.

    The transition is near-zero around chroma ~10 and already close to one
    around chroma ~20.

    Args:
        chroma:
            Chroma values.
        center:
            Midpoint of the transition.
        sharpness:
            Controls how sharp the transition is.

    Returns:
        Array of weights in [0, 1].
    """
    return 0.5 * (1.0 + np.tanh((chroma - center) / sharpness))


def circular_angle_distance(theta: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute circular angular distances between pixel angles and bin centers.

    Args:
        theta:
            Array of shape (N,) with angles in radians.
        centers:
            Array of shape (K,) with bin-center angles in radians.

    Returns:
        Array of shape (N, K) with absolute circular distances in radians.
    """
    delta = theta[:, None] - centers[None, :]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(delta)


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

    Each pixel inside the mask is softly distributed over all hue bins using a
    circular Gaussian kernel. The contribution of each pixel is additionally
    weighted by a soft chroma weight. The final histogram is normalized by the
    full mask area, not by the number of colored pixels.

    Args:
        image:
            BGR uint8 image.
        mask:
            Binary working mask of shape (H, W).
        num_bins:
            Number of circular hue bins.
        sigma_bins:
            Gaussian sigma measured in bin units.
        chroma_center:
            Chroma midpoint for soft weighting.
        chroma_sharpness:
            Chroma transition sharpness.

    Returns:
        Array of shape (num_bins,) with soft color fractions.
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


def directional_color_ratio(
    hist1: np.ndarray,
    hist2: np.ndarray,
    min_fraction: float = 0.20,
) -> float:
    """
    Compute the worst retained ratio from histogram 1 to histogram 2.

    Only bins with mass >= min_fraction in hist1 are checked. For each such bin:
        ratio_i = hist2[i] / hist1[i]

    The returned value is the minimum ratio over all significant bins.

    If hist1 has no significant bins, the function returns 1.0.

    Args:
        hist1:
            Source histogram.
        hist2:
            Target histogram.
        min_fraction:
            Minimum source-bin mass required for checking.

    Returns:
        Worst retained ratio.
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
    Full color-comparison pipeline.

    Steps:
        - crop masks by keypoints
        - remove the top part
        - build soft circular hue histograms
        - compute directional ratios in both directions

    Args:
        image1:
            First BGR image.
        mask1:
            First binary mask.
        src_pts:
            Matched keypoints in image1 coordinates.
        image2:
            Second BGR image.
        mask2:
            Second binary mask.
        dst_pts:
            Matched keypoints in image2 coordinates.
        top_fraction:
            Top part of the mask to remove.
        num_bins:
            Number of hue bins.
        sigma_bins:
            Gaussian sigma in bin units.
        chroma_center:
            Midpoint of the soft chroma transition.
        chroma_sharpness:
            Sharpness of the soft chroma transition.
        min_fraction:
            Minimum bin mass required for directional ratio checking.

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
    work_mask1 = prepare_color_mask(mask1, src_pts, top_fraction=top_fraction)
    work_mask2 = prepare_color_mask(mask2, dst_pts, top_fraction=top_fraction)

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

    ratio_12 = directional_color_ratio(hist1, hist2, min_fraction=min_fraction)
    ratio_21 = directional_color_ratio(hist2, hist1, min_fraction=min_fraction)
    ratio_min = min(ratio_12, ratio_21)

    return ratio_12, ratio_21, ratio_min, hist1, hist2


ratio_12, ratio_21, ratio_min, hist1, hist2 = compare_truck_color_ratio(
    image1=img1,
    mask1=mask1,
    src_pts=src_pts,
    image2=img2,
    mask2=mask2,
    dst_pts=dst_pts,
    top_fraction=0.45,
    num_bins=16,
    sigma_bins=0.9,
    chroma_center=15.0,
    chroma_sharpness=3.0,
    min_fraction=0.20,
)

print("ratio 1->2:", ratio_12)
print("ratio 2->1:", ratio_21)
print("final ratio:", ratio_min)