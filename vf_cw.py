import numpy as np


def chroma_weight(
    chroma: np.ndarray,
    low_center: float | None = None,
    low_sharpness: float | None = None,
    high_center: float | None = None,
    high_sharpness: float | None = None,
) -> np.ndarray:
    """
    Universal soft weight over chroma.

    Supports three modes:
        1. lower-bound only  -> keeps large chroma
        2. upper-bound only  -> keeps small chroma
        3. band-pass         -> keeps chroma inside a soft interval

    Args:
        chroma:
            Array of chroma values.
        low_center:
            Center of the lower soft boundary. Values above it are kept more.
            If None, the lower boundary is disabled.
        low_sharpness:
            Sharpness of the lower soft boundary.
        high_center:
            Center of the upper soft boundary. Values below it are kept more.
            If None, the upper boundary is disabled.
        high_sharpness:
            Sharpness of the upper soft boundary.

    Returns:
        Array of weights in [0, 1].
    """
    weight = np.ones_like(chroma, dtype=np.float32)

    if low_center is not None and low_sharpness is not None:
        lower = 0.5 * (1.0 + np.tanh((chroma - low_center) / low_sharpness))
        weight *= lower.astype(np.float32)

    if high_center is not None and high_sharpness is not None:
        upper = 0.5 * (1.0 - np.tanh((chroma - high_center) / high_sharpness))
        weight *= upper.astype(np.float32)

    return weight


def strong_color_weight(
    chroma: np.ndarray,
    center: float = 15.0,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Soft weight for strongly saturated colors.
    """
    return chroma_weight(
        chroma=chroma,
        low_center=center,
        low_sharpness=sharpness,
    )


def neutral_weight(
    chroma: np.ndarray,
    center: float = 5.0,
    sharpness: float = 0.6,
) -> np.ndarray:
    """
    Soft weight for near-neutral pixels.
    """
    return chroma_weight(
        chroma=chroma,
        high_center=center,
        high_sharpness=sharpness,
    )


def weak_color_weight(
    chroma: np.ndarray,
    low_center: float = 5.0,
    low_sharpness: float = 0.6,
    high_center: float = 15.0,
    high_sharpness: float = 3.0,
) -> np.ndarray:
    """
    Soft band-pass weight for weakly saturated colors.
    """
    return chroma_weight(
        chroma=chroma,
        low_center=low_center,
        low_sharpness=low_sharpness,
        high_center=high_center,
        high_sharpness=high_sharpness,
    )