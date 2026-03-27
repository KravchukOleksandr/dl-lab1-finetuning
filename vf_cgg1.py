import cv2
import numpy as np


def filter_mask_by_chroma_mode(
    image: np.ndarray,
    mask: np.ndarray,
    chroma_thresh: float = 12.0,
    keep: str = "colored",
) -> np.ndarray:
    """
    Keep only colored or only neutral pixels inside a mask.

    Args:
        image:
            BGR uint8 image.
        mask:
            Binary mask.
        chroma_thresh:
            Threshold separating neutral and colored pixels.
        keep:
            "colored" or "neutral".

    Returns:
        Filtered mask.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    chroma = np.sqrt(a * a + b * b)

    if keep == "colored":
        return ((mask > 0) & (chroma >= chroma_thresh)).astype(np.uint8)

    return ((mask > 0) & (chroma < chroma_thresh)).astype(np.uint8)