from __future__ import annotations

import cv2
import numpy as np


def compute_local_patch_score(
    image1: np.ndarray,
    image2: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    patch_radius: int = 12,
    bad_threshold: float = 18.0,
    top_k_ratio: float = 0.1,
) -> dict[str, float]:
    """
    Compare local color patches around matched keypoints.

    The function extracts paired patches around src/dst keypoints, compares them
    in Lab color space using only a,b channels, and returns robust aggregate
    scores. It is intended to detect local color inconsistencies between two
    geometrically matched truck images.

    Args:
        image1:
            First image in BGR uint8 format.
        image2:
            Second image in BGR uint8 format.
        src_pts:
            Matched keypoints in image1 coordinates, shape (N, 2).
        dst_pts:
            Matched keypoints in image2 coordinates, shape (N, 2).
        patch_radius:
            Half-size of the square patch in pixels.
        bad_threshold:
            Per-patch mean Lab a,b difference threshold above which the patch is
            considered bad.
        top_k_ratio:
            Fraction of worst patches used for the top-k mean score.

    Returns:
        A dictionary with:
            - mean_diff:
                Mean patch difference over all valid matched patches.
            - bad_ratio:
                Fraction of patches with diff > bad_threshold.
            - top_k_mean:
                Mean difference of the worst top-k fraction of patches.
            - num_patches:
                Number of valid compared patches.
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    _, a1, b1 = cv2.split(lab1)
    _, a2, b2 = cv2.split(lab2)

    diffs: list[float] = []

    r = patch_radius

    for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        if x1 - r < 0 or x1 + r >= w1 or y1 - r < 0 or y1 + r >= h1:
            continue

        if x2 - r < 0 or x2 + r >= w2 or y2 - r < 0 or y2 + r >= h2:
            continue

        patch_a1 = a1[y1 - r:y1 + r + 1, x1 - r:x1 + r + 1].astype(np.float32)
        patch_b1 = b1[y1 - r:y1 + r + 1, x1 - r:x1 + r + 1].astype(np.float32)

        patch_a2 = a2[y2 - r:y2 + r + 1, x2 - r:x2 + r + 1].astype(np.float32)
        patch_b2 = b2[y2 - r:y2 + r + 1, x2 - r:x2 + r + 1].astype(np.float32)

        diff_a = np.abs(patch_a1 - patch_a2)
        diff_b = np.abs(patch_b1 - patch_b2)

        patch_diff = float((diff_a.mean() + diff_b.mean()) * 0.5)
        diffs.append(patch_diff)

    if not diffs:
        return {
            "mean_diff": 1e9,
            "bad_ratio": 1.0,
            "top_k_mean": 1e9,
            "num_patches": 0,
        }

    diffs_arr = np.asarray(diffs, dtype=np.float32)
    diffs_sorted = np.sort(diffs_arr)

    k = max(1, int(round(len(diffs_sorted) * top_k_ratio)))
    top_k = diffs_sorted[-k:]

    return {
        "mean_diff": float(diffs_arr.mean()),
        "bad_ratio": float((diffs_arr > bad_threshold).mean()),
        "top_k_mean": float(top_k.mean()),
        "num_patches": int(len(diffs_arr)),
    }