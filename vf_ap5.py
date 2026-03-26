import numpy as np


def check_spatial_coverage(
    points: np.ndarray,
    image_height: int,
    min_total: int,
    min_region: int,
) -> bool:
    """
    Check that matched points cover at least 3 of 4 equal vertical regions.

    The crop is assumed to contain the truck from top to bottom, so only the
    image height is used.

    Regions:
        - 0–25%
        - 25–50%
        - 50–75%
        - 75–100%

    Args:
        points:
            Array of shape (N, 2) with point coordinates in (x, y) format.
        image_height:
            Height of the crop.
        min_total:
            Minimum total number of points.
        min_region:
            Minimum number of points required for a region to count as covered.

    Returns:
        True if at least 3 of the 4 regions are covered, otherwise False.
    """
    if len(points) < min_total:
        return False

    y = points[:, 1]
    h = image_height

    counts = [
        ((y >= 0.00 * h) & (y < 0.25 * h)).sum(),
        ((y >= 0.25 * h) & (y < 0.50 * h)).sum(),
        ((y >= 0.50 * h) & (y < 0.75 * h)).sum(),
        ((y >= 0.75 * h) & (y <= 1.00 * h)).sum(),
    ]

    covered_regions = sum(count >= min_region for count in counts)
    return covered_regions >= 3