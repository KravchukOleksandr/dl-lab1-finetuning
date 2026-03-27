import numpy as np


def crop_mask_by_points(
    mask: np.ndarray,
    points: np.ndarray,
    margin: int = 5,
) -> np.ndarray:
    """
    Crop a binary mask by the bounding box of given points.

    Args:
        mask:
            Binary mask (H, W).
        points:
            Array of shape (N, 2) with (x, y) coordinates.
        margin:
            Optional padding (in pixels) around the bounding box.

    Returns:
        Cropped mask of the same shape.
    """
    x = points[:, 0]
    y = points[:, 1]

    x1 = max(0, int(x.min()) - margin)
    x2 = min(mask.shape[1], int(x.max()) + margin)

    y1 = max(0, int(y.min()) - margin)
    y2 = min(mask.shape[0], int(y.max()) + margin)

    cropped = np.zeros_like(mask)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    return cropped