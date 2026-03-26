from __future__ import annotations

import cv2
import numpy as np


def compute_color_distance(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    src_pts: np.ndarray,
    bins: int = 32,
) -> float:
    """
    Compute color distribution distance between two images using matched keypoints region.

    The function:
    - builds a convex hull over src_pts (in image1 space)
    - restricts both masks to that region
    - compares color distributions in Lab space (a,b channels only)
    - returns Bhattacharyya distance between histograms

    Args:
        image1:
            First image (BGR uint8).
        mask1:
            Binary mask for image1 (same H,W).
        image2:
            Second image (BGR uint8).
        mask2:
            Binary mask for image2 (same H,W).
        src_pts:
            Matched keypoints in image1 coordinates (N, 2).
        bins:
            Number of histogram bins per channel.

    Returns:
        Bhattacharyya distance in [0, 1].
        Larger → stronger color difference.
    """
    h, w = image1.shape[:2]

    # --- build convex hull from keypoints ---
    hull = cv2.convexHull(src_pts.astype(np.float32))
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 1)

    # --- restrict masks ---
    mask1_final = (mask1 > 0) & (hull_mask > 0)
    mask2_final = (mask2 > 0) & (hull_mask > 0)

    # --- convert to Lab ---
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    L1, a1, b1 = cv2.split(lab1)
    L2, a2, b2 = cv2.split(lab2)

    # --- filter valid pixels (remove shadows / highlights) ---
    valid1 = mask1_final & (L1 > 40) & (L1 < 220)
    valid2 = mask2_final & (L2 > 40) & (L2 < 220)

    a1_vals = a1[valid1]
    b1_vals = b1[valid1]

    a2_vals = a2[valid2]
    b2_vals = b2[valid2]

    if len(a1_vals) < 50 or len(a2_vals) < 50:
        return 1.0  # not enough data → treat as different

    # --- build histograms ---
    hist1 = cv2.calcHist(
        [np.stack([a1_vals, b1_vals], axis=1)],
        [0, 1],
        None,
        [bins, bins],
        [0, 256, 0, 256],
    )

    hist2 = cv2.calcHist(
        [np.stack([a2_vals, b2_vals], axis=1)],
        [0, 1],
        None,
        [bins, bins],
        [0, 256, 0, 256],
    )

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # --- compare ---
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return float(dist)