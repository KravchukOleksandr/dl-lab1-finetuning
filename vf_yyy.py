import numpy as np


def cut_top_fraction(mask: np.ndarray, top_fraction: float = 0.45) -> np.ndarray:
    """
    Remove top part of mask.
    """
    h = mask.shape[0]
    out = mask.copy()
    out[: int(h * top_fraction), :] = 0
    return out


def cut_side_fractions(
    mask: np.ndarray,
    left_fraction: float = 0.0,
    right_fraction: float = 0.0,
) -> np.ndarray:
    """
    Remove left and right parts of mask.

    left_fraction and right_fraction are fractions of mask width.
    """
    h, w = mask.shape[:2]
    out = mask.copy()

    left_w = int(w * left_fraction)
    right_w = int(w * right_fraction)

    if left_w > 0:
        out[:, :left_w] = 0

    if right_w > 0:
        out[:, w - right_w:] = 0

    return out


def filter_matched_points_by_masks(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only matched pairs whose endpoints lie inside both masks.
    """
    if len(src_pts) == 0:
        return src_pts, dst_pts

    x1 = np.round(src_pts[:, 0]).astype(int)
    y1 = np.round(src_pts[:, 1]).astype(int)

    x2 = np.round(dst_pts[:, 0]).astype(int)
    y2 = np.round(dst_pts[:, 1]).astype(int)

    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]

    valid = (
        (x1 >= 0) & (x1 < w1) &
        (y1 >= 0) & (y1 < h1) &
        (mask1[y1, x1] > 0) &
        (x2 >= 0) & (x2 < w2) &
        (y2 >= 0) & (y2 < h2) &
        (mask2[y2, x2] > 0)
    )

    return src_pts[valid], dst_pts[valid]


def crop_mask_by_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Crop mask to bbox of given points.
    """
    if len(points) == 0:
        return np.zeros_like(mask)

    x = points[:, 0]
    y = points[:, 1]

    x1 = max(0, int(np.floor(x.min())))
    x2 = min(mask.shape[1], int(np.ceil(x.max())))
    y1 = max(0, int(np.floor(y.min())))
    y2 = min(mask.shape[0], int(np.ceil(y.max())))

    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def prepare_work_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    top_fraction: float = 0.45,
    left_fraction: float = 0.0,
    right_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare working masks.

    Order:
        1) cut top
        2) cut left/right
        3) filter matched points
        4) crop by filtered points
    """
    cut_mask1 = cut_top_fraction(mask1, top_fraction)
    cut_mask2 = cut_top_fraction(mask2, top_fraction)

    cut_mask1 = cut_side_fractions(
        cut_mask1,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
    )
    cut_mask2 = cut_side_fractions(
        cut_mask2,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
    )

    src_pts_f, dst_pts_f = filter_matched_points_by_masks(
        src_pts, dst_pts, cut_mask1, cut_mask2
    )

    work_mask1 = crop_mask_by_points(cut_mask1, src_pts_f)
    work_mask2 = crop_mask_by_points(cut_mask2, dst_pts_f)

    return work_mask1, work_mask2, src_pts_f, dst_pts_f