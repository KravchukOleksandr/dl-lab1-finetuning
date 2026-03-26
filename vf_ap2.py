import numpy as np


def check_spatial_coverage(
    points: np.ndarray,
    mask: np.ndarray,
    min_total: int,
    min_top: int,
    min_mid: int,
    min_bottom: int,
) -> bool:
    """
    Check that points are sufficiently distributed across key vertical regions.

    Regions (based on mask bbox height):
        - top:    0–20%
        - mid:    50–75%   (below windshield)
        - bottom: 75–100%

    Args:
        points: (N, 2) array of (x, y)
        mask: binary mask
        min_total: minimum total number of points
        min_top: minimum points in top region
        min_mid: minimum points in mid region
        min_bottom: minimum points in bottom region

    Returns:
        bool
    """
    if len(points) < min_total:
        return False

    ys = np.where(mask > 0)[0]
    y_top = ys.min()
    y_bottom = ys.max()
    h = y_bottom - y_top

    y = points[:, 1]

    top_thr = y_top + 0.2 * h
    mid_low = y_top + 0.5 * h
    mid_high = y_top + 0.75 * h

    top = (y >= y_top) & (y < top_thr)
    mid = (y >= mid_low) & (y < mid_high)
    bottom = (y >= mid_high) & (y <= y_bottom)

    return (
        top.sum() >= min_top and
        mid.sum() >= min_mid and
        bottom.sum() >= min_bottom
    )