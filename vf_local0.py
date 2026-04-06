from __future__ import annotations

import cv2
import numpy as np


def make_self_attention_augmentation(
    image: np.ndarray,
    blur_ksize: int = 9,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    """
    Deterministic geometry-preserving augmentation for self matching.
    """
    return cv2.GaussianBlur(image, (blur_ksize, blur_ksize), blur_sigma)


def filter_points_by_mask(
    points: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Keep only points that lie inside the mask.
    """
    if len(points) == 0:
        return points

    x = np.round(points[:, 0]).astype(np.int32)
    y = np.round(points[:, 1]).astype(np.int32)

    h, w = mask.shape[:2]
    valid = (
        (x >= 0) & (x < w) &
        (y >= 0) & (y < h) &
        (mask[y, x] > 0)
    )
    return points[valid]


def filter_matched_points_by_masks(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    src_mask: np.ndarray,
    dst_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only matched pairs whose endpoints lie inside both masks.
    """
    if len(src_pts) == 0:
        return src_pts, dst_pts

    x1 = np.round(src_pts[:, 0]).astype(np.int32)
    y1 = np.round(src_pts[:, 1]).astype(np.int32)

    x2 = np.round(dst_pts[:, 0]).astype(np.int32)
    y2 = np.round(dst_pts[:, 1]).astype(np.int32)

    h1, w1 = src_mask.shape[:2]
    h2, w2 = dst_mask.shape[:2]

    valid = (
        (x1 >= 0) & (x1 < w1) &
        (y1 >= 0) & (y1 < h1) &
        (src_mask[y1, x1] > 0) &
        (x2 >= 0) & (x2 < w2) &
        (y2 >= 0) & (y2 < h2) &
        (dst_mask[y2, x2] > 0)
    )

    return src_pts[valid], dst_pts[valid]


def make_count_map(
    points: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize points into a 2D count map.
    """
    h, w = shape
    count_map = np.zeros((h, w), dtype=np.int32)

    if len(points) == 0:
        return count_map

    x = np.round(points[:, 0]).astype(np.int32)
    y = np.round(points[:, 1]).astype(np.int32)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[valid]
    y = y[valid]

    np.add.at(count_map, (y, x), 1)
    return count_map


def make_integral_image(image: np.ndarray) -> np.ndarray:
    """
    Build integral image with one-pixel padding.
    """
    return cv2.integral(image)


def rect_sum(
    integral: np.ndarray,
    rect: tuple[int, int, int, int],
) -> float:
    """
    Sum values inside rect = (x1, y1, x2, y2), x2/y2 exclusive.
    """
    x1, y1, x2, y2 = rect
    return float(
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )


def rect_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """
    IoU of two axis-aligned rectangles.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union == 0:
        return 0.0

    return inter / union


def best_square_around_center(
    mask_integral: np.ndarray,
    mask_shape: tuple[int, int],
    cx: int,
    cy: int,
    size: int,
    shift_radius: int = 12,
    min_mask_coverage: float = 0.90,
) -> tuple[int, int, int, int] | None:
    """
    Find the best shifted square around a center so that it stays inside the mask as much as possible.
    """
    h, w = mask_shape
    half = size // 2
    target_area = size * size

    best_rect = None
    best_score = -1.0
    best_shift = 10**9

    for dy in range(-shift_radius, shift_radius + 1):
        for dx in range(-shift_radius, shift_radius + 1):
            x1 = int(np.clip(cx - half + dx, 0, w - size))
            y1 = int(np.clip(cy - half + dy, 0, h - size))
            x2 = x1 + size
            y2 = y1 + size

            rect = (x1, y1, x2, y2)
            mask_pixels = rect_sum(mask_integral, rect)

            if mask_pixels < min_mask_coverage * target_area:
                continue

            shift_score = dx * dx + dy * dy
            if mask_pixels > best_score or (mask_pixels == best_score and shift_score < best_shift):
                best_rect = rect
                best_score = mask_pixels
                best_shift = shift_score

    return best_rect


def select_discriminative_regions(
    self_points: np.ndarray,
    cross_points: np.ndarray,
    mask: np.ndarray,
    k: int = 6,
    size: int = 56,
    candidate_step: int = 8,
    shift_radius: int = 12,
    min_mask_coverage: float = 0.90,
    cross_weight: float = 1.5,
    max_iou: float = 0.10,
) -> list[dict]:
    """
    Select k non-overlapping squares with many self points and few cross points.
    """
    h, w = mask.shape[:2]

    mask_map = (mask > 0).astype(np.uint8)
    self_map = make_count_map(self_points, (h, w))
    cross_map = make_count_map(cross_points, (h, w))

    mask_integral = make_integral_image(mask_map)
    self_integral = make_integral_image(self_map)
    cross_integral = make_integral_image(cross_map)

    ys, xs = np.where(mask_map > 0)
    if len(xs) == 0:
        return []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    candidates: list[dict] = []

    for cy in range(y_min, y_max + 1, candidate_step):
        for cx in range(x_min, x_max + 1, candidate_step):
            if mask_map[cy, cx] == 0:
                continue

            rect = best_square_around_center(
                mask_integral=mask_integral,
                mask_shape=(h, w),
                cx=cx,
                cy=cy,
                size=size,
                shift_radius=shift_radius,
                min_mask_coverage=min_mask_coverage,
            )
            if rect is None:
                continue

            self_count = rect_sum(self_integral, rect)
            cross_count = rect_sum(cross_integral, rect)

            if self_count <= 0:
                continue

            score = self_count - cross_weight * cross_count

            candidates.append(
                {
                    "rect": rect,
                    "self_count": int(self_count),
                    "cross_count": int(cross_count),
                    "score": float(score),
                }
            )

    candidates.sort(key=lambda x: (x["score"], x["self_count"]), reverse=True)

    selected: list[dict] = []
    for cand in candidates:
        overlaps = any(rect_iou(cand["rect"], prev["rect"]) > max_iou for prev in selected)
        if overlaps:
            continue

        selected.append(cand)
        if len(selected) == k:
            break

    return selected


def estimate_global_affine(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray | None:
    """
    Estimate a global affine transform from cross matches.
    """
    if len(src_pts) < 3:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return M


def transform_rect(
    rect: tuple[int, int, int, int],
    M: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Transform an axis-aligned rectangle and return its enclosing axis-aligned bbox.
    """
    x1, y1, x2, y2 = rect
    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )

    transformed = cv2.transform(corners[None, :, :], M)[0]

    h, w = out_shape
    tx1 = int(np.floor(np.clip(transformed[:, 0].min(), 0, w - 1)))
    ty1 = int(np.floor(np.clip(transformed[:, 1].min(), 0, h - 1)))
    tx2 = int(np.ceil(np.clip(transformed[:, 0].max(), 0, w - 1)))
    ty2 = int(np.ceil(np.clip(transformed[:, 1].max(), 0, h - 1)))

    return tx1, ty1, tx2, ty2


def find_discriminative_patch_regions(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    extractor: Any,
    matcher: Any,
    device: str = "cpu",
    k: int = 6,
    size: int = 56,
    blur_ksize: int = 9,
    blur_sigma: float = 2.0,
    candidate_step: int = 8,
    shift_radius: int = 12,
    min_mask_coverage: float = 0.90,
    cross_weight: float = 1.5,
    max_iou: float = 0.10,
) -> dict[str, Any]:
    """
    Find k discriminative square regions on truck 1 and transfer them to truck 2.

    The regions are selected to have:
        - many self points on truck 1
        - few cross points between truck 1 and truck 2
    """
    aug1 = make_self_attention_augmentation(
        image=image1,
        blur_ksize=blur_ksize,
        blur_sigma=blur_sigma,
    )

    feats1 = extract_image_features(extractor=extractor, frame=image1, device=device)
    feats1_aug = extract_image_features(extractor=extractor, frame=aug1, device=device)
    feats2 = extract_image_features(extractor=extractor, frame=image2, device=device)

    _, self_src_pts, self_dst_pts = match_images(
        matcher=matcher,
        feats0=feats1,
        feats1=feats1_aug,
    )
    self_src_pts = filter_points_by_mask(self_src_pts, mask1)
    self_dst_pts = filter_points_by_mask(self_dst_pts, mask1)

    _, cross_src_pts, cross_dst_pts = match_images(
        matcher=matcher,
        feats0=feats1,
        feats1=feats2,
    )
    cross_src_pts, cross_dst_pts = filter_matched_points_by_masks(
        src_pts=cross_src_pts,
        dst_pts=cross_dst_pts,
        src_mask=mask1,
        dst_mask=mask2,
    )

    regions1 = select_discriminative_regions(
        self_points=self_src_pts,
        cross_points=cross_src_pts,
        mask=mask1,
        k=k,
        size=size,
        candidate_step=candidate_step,
        shift_radius=shift_radius,
        min_mask_coverage=min_mask_coverage,
        cross_weight=cross_weight,
        max_iou=max_iou,
    )

    M = estimate_global_affine(cross_src_pts, cross_dst_pts)

    regions2: list[dict] = []
    if M is not None:
        for region in regions1:
            rect2 = transform_rect(
                rect=region["rect"],
                M=M,
                out_shape=image2.shape[:2],
            )
            regions2.append(
                {
                    "rect": rect2,
                    "score": region["score"],
                    "self_count": region["self_count"],
                    "cross_count": region["cross_count"],
                }
            )

    return {
        "aug1": aug1,
        "self_src_pts": self_src_pts,
        "self_dst_pts": self_dst_pts,
        "cross_src_pts": cross_src_pts,
        "cross_dst_pts": cross_dst_pts,
        "regions1": regions1,
        "regions2": regions2,
        "affine": M,
    }


def draw_self_attention_points(
    image: np.ndarray,
    aug_image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Visualize self-attention matches: original on the left, augmented on the right.

    Returns an RGB image.
    """
    left = image.copy()
    right = aug_image.copy()

    for p in src_pts:
        x, y = np.round(p).astype(int)
        cv2.circle(left, (x, y), radius, (0, 255, 0), -1)

    for p in dst_pts:
        x, y = np.round(p).astype(int)
        cv2.circle(right, (x, y), radius, (0, 255, 0), -1)

    canvas = np.concatenate([left, right], axis=1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_regions_transfer(
    image1: np.ndarray,
    image2: np.ndarray,
    regions1: list[dict],
    regions2: list[dict],
    thickness: int = 2,
) -> np.ndarray:
    """
    Visualize selected regions on truck 1 and their transferred regions on truck 2.

    Returns an RGB image.
    """
    left = image1.copy()
    right = image2.copy()

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, region in enumerate(regions1):
        color = palette[i % len(palette)]
        x1, y1, x2, y2 = region["rect"]
        cv2.rectangle(left, (x1, y1), (x2, y2), color, thickness)

    for i, region in enumerate(regions2):
        color = palette[i % len(palette)]
        x1, y1, x2, y2 = region["rect"]
        cv2.rectangle(right, (x1, y1), (x2, y2), color, thickness)

    canvas = np.concatenate([left, right], axis=1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)