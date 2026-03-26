import numpy as np


def check_spatial_coverage(
    points: np.ndarray,
    image_height: int,
    min_total: int,
    min_region: int,
) -> bool:
    """
    Check that matched points cover the key vertical parts of a truck crop.

    The crop is assumed to contain the truck from top to bottom, so only the
    image height is used.

    Regions:
        - top:    0–25%
        - middle: 50–75%
        - bottom: 75–100%

    The 25–50% band is intentionally excluded as a rough approximation of the
    windshield area.

    Args:
        points:
            Array of shape (N, 2) with point coordinates in (x, y) format.
        image_height:
            Height of the crop.
        min_total:
            Minimum total number of points.
        min_region:
            Minimum number of points required in each checked region.

    Returns:
        True if the point set is sufficiently distributed, otherwise False.
    """
    if len(points) < min_total:
        return False

    y = points[:, 1]

    top_end = 0.25 * image_height
    mid_start = 0.50 * image_height
    mid_end = 0.75 * image_height
    bottom_start = 0.75 * image_height

    top = ((y >= 0) & (y < top_end)).sum()
    middle = ((y >= mid_start) & (y < mid_end)).sum()
    bottom = ((y >= bottom_start) & (y <= image_height)).sum()

    return top >= min_region and middle >= min_region and bottom >= min_region