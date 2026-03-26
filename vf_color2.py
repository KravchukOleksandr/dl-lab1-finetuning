from __future__ import annotations

import cv2
import numpy as np


def compute_color_distance(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    bins: int = 32,
) -> float:
    """
    Compute color distribution distance between two matched regions.

    The function:
    - builds a convex hull over src_pts in image1
    - builds a convex hull over dst_pts in image2
    - restricts both masks to these hull regions
    - compares color distributions in Lab space using only a,b channels
    - returns Bhattacharyya distance between normalized histograms

    Args:
        image1:
            First image in BGR uint8 format.
        mask1:
            Binary mask for the first image.
        image2:
            Second image in BGR uint8 format.
        mask2:
            Binary mask for the second image.
        src_pts:
            Matched keypoints in image1 coordinates, shape (N, 2).
        dst_pts:
            Matched keypoints in image2 coordinates, shape (N, 2).
        bins:
            Number of histogram bins per a/b channel.

    Returns:
        Bhattacharyya distance in the range [0, 1].
        Larger values indicate stronger color mismatch.
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    hull1 = cv2.convexHull(src_pts.astype(np.float32))
    hull2 = cv2.convexHull(dst_pts.astype(np.float32))

    hull_mask1 = np.zeros((h1, w1), dtype=np.uint8)
    hull_mask2 = np.zeros((h2, w2), dtype=np.uint8)

    cv2.fillConvexPoly(hull_mask1, hull1.astype(np.int32), 1)
    cv2.fillConvexPoly(hull_mask2, hull2.astype(np.int32), 1)

    mask1_final = (mask1 > 0) & (hull_mask1 > 0)
    mask2_final = (mask2 > 0) & (hull_mask2 > 0)

    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    l1, a1, b1 = cv2.split(lab1)
    l2, a2, b2 = cv2.split(lab2)

    valid1 = mask1_final & (l1 > 40) & (l1 < 220)
    valid2 = mask2_final & (l2 > 40) & (l2 < 220)

    ab1 = np.stack([a1[valid1], b1[valid1]], axis=1).astype(np.float32)
    ab2 = np.stack([a2[valid2], b2[valid2]], axis=1).astype(np.float32)

    if len(ab1) < 50 or len(ab2) < 50:
        return 1.0

    hist1 = cv2.calcHist([ab1], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    hist2 = cv2.calcHist([ab2], [0, 1], None, [bins, bins], [0, 256, 0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return float(dist)