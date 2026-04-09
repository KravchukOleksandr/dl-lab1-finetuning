from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import cv2
import numpy as np


# ============================================================
# EXTERNAL FUNCTIONS EXPECTED TO EXIST
# ============================================================
# You already have these functions implemented.
#
# Expected behavior:
#   compute_color_hist(image1, mask1, image2, mask2,
#                      num_bins, sigma_bins,
#                      low_chroma_center, low_chroma_sharpness,
#                      high_chroma_center, high_chroma_sharpness)
#       -> hist1, hist2
#
#   compute_brightness_hist(image1, mask1, image2, mask2,
#                           num_bins, sigma_bins,
#                           low_chroma_center, low_chroma_sharpness,
#                           high_chroma_center, high_chroma_sharpness)
#       -> hist1, hist2
#
# The code below assumes those functions return normalized histograms.
# ============================================================


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ColorChannelResult:
    active: bool
    score: Optional[float]
    active_indices: List[int]


@dataclass
class PatchColorDescriptor:
    W_s: np.ndarray
    active_s: List[int]
    W_w: np.ndarray
    active_w: List[int]
    W_b: np.ndarray
    active_b: List[int]


@dataclass
class PatchColorScores:
    strong: ColorChannelResult
    weak: ColorChannelResult
    brightness: ColorChannelResult


@dataclass
class CenterScores:
    center1: Tuple[int, int]
    pred_center2: Tuple[float, float]

    best_R_strong: Optional[float]
    best_R_weak: Optional[float]
    best_R_brightness: Optional[float]
    best_d_hog: Optional[float]

    best_delta_strong: Optional[Tuple[int, int]]
    best_delta_weak: Optional[Tuple[int, int]]
    best_delta_brightness: Optional[Tuple[int, int]]
    best_delta_hog: Optional[Tuple[int, int]]

    strong_active: bool
    weak_active: bool
    brightness_active: bool

    num_valid_candidates: int


# ============================================================
# BASIC GEOMETRY
# ============================================================

def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return bounding box of non-zero mask pixels as (x, y, w, h).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def estimate_global_affine(
    pts1: np.ndarray,
    pts2: np.ndarray,
    ransac_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate global affine transform from pts1 to pts2.

    Returns:
        A: 2x2 matrix
        t: 2-vector
        inliers: boolean mask
    """
    if pts1.shape[0] < 3:
        raise ValueError("Need at least 3 matches to estimate affine")

    M, inliers = cv2.estimateAffine2D(
        pts1.astype(np.float32),
        pts2.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.99,
        refineIters=10,
    )

    if M is None:
        raise RuntimeError("estimateAffine2D failed")

    A = M[:, :2].astype(np.float32)
    t = M[:, 2].astype(np.float32)
    inliers = inliers.ravel().astype(bool)

    return A, t, inliers


def affine_transfer_errors(
    pts1: np.ndarray,
    pts2: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean transfer error for matched points.
    """
    pred = pts1 @ A.T + t[None, :]
    err = np.linalg.norm(pred - pts2, axis=1)
    return err


def auto_search_radius_from_affine_errors(
    pts1: np.ndarray,
    pts2: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    q: float = 0.90,
    scale: float = 2.0,
    r_min: int = 8,
    r_max: int = 30,
) -> int:
    """
    Set search radius from affine transfer error statistics.
    """
    err = affine_transfer_errors(pts1, pts2, A, t)
    e_q = float(np.quantile(err, q))
    radius = int(round(scale * e_q))
    radius = max(r_min, min(r_max, radius))
    return radius


# ============================================================
# PATCH CENTER GENERATION
# ============================================================

def generate_positions_1d(start: int, end: int, step: int) -> List[int]:
    """
    Generate 1D grid positions and force inclusion of the last valid position.
    """
    if start > end:
        return []
    if step <= 0:
        raise ValueError("step must be positive")

    vals = list(range(start, end + 1, step))
    if vals[-1] != end:
        vals.append(end)
    return vals


def generate_patch_centers_from_mask_bbox(
    mask: np.ndarray,
    patch_size: int = 48,
    step_diag_frac: float = 0.06,
) -> List[Tuple[int, int]]:
    """
    Generate patch centers on a regular grid inside the bounding box of mask.

    The center step is:
        step = round(step_diag_frac * diagonal_of_bbox)

    Centers are chosen so that the square patch fits fully inside
    the image and inside the bounding box along x and y.
    """
    H, W = mask.shape[:2]
    x, y, w, h = bbox_from_mask(mask)

    diag = math.sqrt(w * w + h * h)
    step = max(1, int(round(step_diag_frac * diag)))

    half = patch_size // 2

    x_start = max(x + half, half)
    x_end = min(x + w - 1 - half, W - 1 - half)

    y_start = max(y + half, half)
    y_end = min(y + h - 1 - half, H - 1 - half)

    xs = generate_positions_1d(x_start, x_end, step)
    ys = generate_positions_1d(y_start, y_end, step)

    centers = [(cx, cy) for cy in ys for cx in xs]
    return centers


# ============================================================
# PATCH EXTRACTION
# ============================================================

def build_rect_patch(
    image: np.ndarray,
    center: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """
    Extract a square patch from image around center.
    """
    return cv2.getRectSubPix(
        image,
        patchSize=(patch_size, patch_size),
        center=(float(center[0]), float(center[1])),
    )


def candidate_patch_fully_inside_image(
    center1: Tuple[int, int],
    A: np.ndarray,
    t: np.ndarray,
    delta: Tuple[int, int],
    patch_size: int,
    image_shape: Tuple[int, int],
) -> bool:
    """
    Check whether the affine-transferred patch from image1,
    after applying delta, lies fully inside image2.
    """
    H, W = image_shape[:2]
    half = patch_size // 2

    corners_local = np.array([
        [-half, -half],
        [half - 1, -half],
        [-half, half - 1],
        [half - 1, half - 1],
    ], dtype=np.float32)

    c1 = np.array(center1, dtype=np.float32)[None, :]
    pts1 = c1 + corners_local
    pts2 = pts1 @ A.T + t[None, :] + np.array(delta, dtype=np.float32)[None, :]

    xs = pts2[:, 0]
    ys = pts2[:, 1]

    return (
        xs.min() >= 0 and ys.min() >= 0 and
        xs.max() <= W - 1 and ys.max() <= H - 1
    )


def build_canonical_candidate_patch(
    image2: np.ndarray,
    center1: Tuple[int, int],
    A: np.ndarray,
    t: np.ndarray,
    delta: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """
    Build canonical candidate patch Q_delta(u, v):

        Q_delta(u, v) = image2( A * (center1 + (u, v)) + t + delta )

    This resamples the affine-deformed patch in image2 back into
    the square canonical grid of size patch_size x patch_size.
    """
    S = patch_size
    half = S // 2

    u = np.arange(S, dtype=np.float32) - half
    v = np.arange(S, dtype=np.float32) - half
    uu, vv = np.meshgrid(u, v)

    x1 = center1[0] + uu
    y1 = center1[1] + vv

    x2 = A[0, 0] * x1 + A[0, 1] * y1 + t[0] + float(delta[0])
    y2 = A[1, 0] * x1 + A[1, 1] * y1 + t[1] + float(delta[1])

    Q = cv2.remap(
        image2,
        x2.astype(np.float32),
        y2.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Q


def generate_deltas(radius: int, step: int = 2) -> List[Tuple[int, int]]:
    """
    Generate integer deltas inside a circular search radius.
    """
    deltas = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx * dx + dy * dy <= radius * radius:
                deltas.append((dx, dy))
    return deltas


# ============================================================
# COLOR DESCRIPTORS
# ============================================================

def windowize_circular(hist: np.ndarray) -> np.ndarray:
    """
    Circular 3-bin window aggregation for hue-like histograms.
    """
    n = len(hist)
    W = np.zeros_like(hist, dtype=np.float32)
    for i in range(n):
        W[i] = hist[(i - 1) % n] + hist[i] + hist[(i + 1) % n]
    return W


def windowize_linear(hist: np.ndarray) -> np.ndarray:
    """
    Linear 3-bin window aggregation for brightness-like histograms.
    """
    n = len(hist)
    W = np.zeros_like(hist, dtype=np.float32)
    for i in range(n):
        left = hist[max(i - 1, 0)]
        mid = hist[i]
        right = hist[min(i + 1, n - 1)]
        W[i] = left + mid + right
    return W


def get_active_indices(W_ref: np.ndarray, min_fraction: float) -> List[int]:
    """
    Return indices of histogram windows whose mass is at least min_fraction.
    """
    return [i for i in range(len(W_ref)) if float(W_ref[i]) >= float(min_fraction)]


def directed_ratios(
    W_ref: np.ndarray,
    W_cmp: np.ndarray,
    active_indices: List[int],
    eps: float = 1e-8,
) -> List[float]:
    """
    Directed ratio:
        ratio_i = min(1, W_cmp[i] / W_ref[i])

    The direction is important:
        reference patch -> candidate patch
    """
    ratios = []
    for i in active_indices:
        r = float(W_cmp[i]) / (float(W_ref[i]) + eps)
        r = min(1.0, r)
        ratios.append(r)
    return ratios


def aggregate_ratios(
    ratios: List[float],
    mode: str = "softmin",
    softmin_p: int = 8,
    eps: float = 1e-8,
) -> Optional[float]:
    """
    Aggregate directed ratios into a single score.

    mode:
        - "min": hard minimum
        - "softmin": soft minimum
    """
    if len(ratios) == 0:
        return None

    if mode == "min":
        return float(min(ratios))

    if mode == "softmin":
        acc = 0.0
        for r in ratios:
            acc += max(r, eps) ** (-softmin_p)
        acc /= len(ratios)
        return float(acc ** (-1.0 / softmin_p))

    raise ValueError(f"Unknown ratio mode: {mode}")


def build_patch_color_descriptor(
    patch_bgr: np.ndarray,
    strong_params: Dict,
    weak_params: Dict,
    brightness_params: Dict,
) -> PatchColorDescriptor:
    """
    Build color descriptor for one patch.

    This function assumes compute_color_hist / compute_brightness_hist
    work on two images. To reuse them, the same patch is passed twice,
    and only the first returned histogram is used.
    """
    S = patch_bgr.shape[0]
    patch_mask = np.ones((S, S), dtype=np.uint8)

    # Strong color histogram
    hist_s, _ = compute_color_hist(
        patch_bgr, patch_mask,
        patch_bgr, patch_mask,
        strong_params["num_bins"],
        strong_params["sigma_bins"],
        strong_params.get("low_chroma_center"),
        strong_params.get("low_chroma_sharpness"),
        strong_params.get("high_chroma_center"),
        strong_params.get("high_chroma_sharpness"),
    )
    W_s = windowize_circular(hist_s)
    active_s = get_active_indices(W_s, strong_params["min_fraction"])

    # Weak color histogram
    hist_w, _ = compute_color_hist(
        patch_bgr, patch_mask,
        patch_bgr, patch_mask,
        weak_params["num_bins"],
        weak_params["sigma_bins"],
        weak_params.get("low_chroma_center"),
        weak_params.get("low_chroma_sharpness"),
        weak_params.get("high_chroma_center"),
        weak_params.get("high_chroma_sharpness"),
    )
    W_w = windowize_circular(hist_w)
    active_w = get_active_indices(W_w, weak_params["min_fraction"])

    # Brightness histogram on low-chroma pixels
    hist_b, _ = compute_brightness_hist(
        patch_bgr, patch_mask,
        patch_bgr, patch_mask,
        brightness_params["num_bins"],
        brightness_params["sigma_bins"],
        brightness_params.get("low_chroma_center"),
        brightness_params.get("low_chroma_sharpness"),
        brightness_params.get("high_chroma_center"),
        brightness_params.get("high_chroma_sharpness"),
    )
    W_b = windowize_linear(hist_b)
    active_b = get_active_indices(W_b, brightness_params["min_fraction"])

    return PatchColorDescriptor(
        W_s=W_s,
        active_s=active_s,
        W_w=W_w,
        active_w=active_w,
        W_b=W_b,
        active_b=active_b,
    )


def compare_patch_color_descriptors(
    desc_ref: PatchColorDescriptor,
    desc_cmp: PatchColorDescriptor,
    ratio_mode: str = "softmin",
    strong_softmin_p: int = 8,
    weak_softmin_p: int = 8,
    brightness_softmin_p: int = 8,
) -> PatchColorScores:
    """
    Compare two patch color descriptors in directed mode:
        reference patch -> candidate patch
    """
    ratios_s = directed_ratios(desc_ref.W_s, desc_cmp.W_s, desc_ref.active_s)
    ratios_w = directed_ratios(desc_ref.W_w, desc_cmp.W_w, desc_ref.active_w)
    ratios_b = directed_ratios(desc_ref.W_b, desc_cmp.W_b, desc_ref.active_b)

    R_strong = aggregate_ratios(
        ratios_s, mode=ratio_mode, softmin_p=strong_softmin_p
    )
    R_weak = aggregate_ratios(
        ratios_w, mode=ratio_mode, softmin_p=weak_softmin_p
    )
    R_brightness = aggregate_ratios(
        ratios_b, mode=ratio_mode, softmin_p=brightness_softmin_p
    )

    return PatchColorScores(
        strong=ColorChannelResult(
            active=(len(desc_ref.active_s) > 0),
            score=R_strong,
            active_indices=desc_ref.active_s,
        ),
        weak=ColorChannelResult(
            active=(len(desc_ref.active_w) > 0),
            score=R_weak,
            active_indices=desc_ref.active_w,
        ),
        brightness=ColorChannelResult(
            active=(len(desc_ref.active_b) > 0),
            score=R_brightness,
            active_indices=desc_ref.active_b,
        ),
    )


# ============================================================
# HOG DESCRIPTOR
# ============================================================

def extract_L_channel(patch_bgr: np.ndarray) -> np.ndarray:
    """
    Extract L channel in [0, 100] from BGR patch.
    """
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    return L


def add_soft_orientation_vote(hist: np.ndarray, theta: float, mag: float) -> None:
    """
    Soft vote into two neighboring orientation bins.

    theta is expected in [0, pi).
    """
    n = len(hist)
    bin_pos = (theta / math.pi) * n
    i0 = int(math.floor(bin_pos)) % n
    frac = bin_pos - math.floor(bin_pos)
    i1 = (i0 + 1) % n

    hist[i0] += mag * (1.0 - frac)
    hist[i1] += mag * frac


def compute_patch_hog_descriptor(
    patch_bgr: np.ndarray,
    num_angle_bins: int = 8,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """
    Build HOG descriptor:
        - one global orientation histogram
        - one coarse 2x2 spatial grid

    This is intentionally coarse for better stability.
    """
    L = extract_L_channel(patch_bgr)
    if gaussian_sigma > 0:
        L = cv2.GaussianBlur(L, (0, 0), gaussian_sigma)

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) % np.pi

    H, W = L.shape
    half_h = H // 2
    half_w = W // 2

    desc_parts = []

    # Global histogram
    h_global = np.zeros(num_angle_bins, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            add_soft_orientation_vote(h_global, float(ang[y, x]), float(mag[y, x]))
    h_global /= (np.linalg.norm(h_global) + 1e-8)
    desc_parts.append(h_global)

    # Coarse 2x2 grid
    cells = [
        (0, half_h, 0, half_w),
        (0, half_h, half_w, W),
        (half_h, H, 0, half_w),
        (half_h, H, half_w, W),
    ]

    for y0, y1, x0, x1 in cells:
        h_cell = np.zeros(num_angle_bins, dtype=np.float32)
        for y in range(y0, y1):
            for x in range(x0, x1):
                add_soft_orientation_vote(h_cell, float(ang[y, x]), float(mag[y, x]))
        h_cell /= (np.linalg.norm(h_cell) + 1e-8)
        desc_parts.append(h_cell)

    desc = np.concatenate(desc_parts, axis=0)
    desc /= (np.linalg.norm(desc) + 1e-8)
    return desc.astype(np.float32)


def compare_hog_descriptors(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Cosine distance for normalized HOG descriptors.
    """
    return float(1.0 - np.dot(desc1, desc2))


# ============================================================
# SEARCH FOR ONE CENTER
# ============================================================

def search_scores_for_center(
    image1: np.ndarray,
    image2: np.ndarray,
    center1: Tuple[int, int],
    A: np.ndarray,
    t: np.ndarray,
    patch_size: int,
    radius: int,
    strong_params: Dict,
    weak_params: Dict,
    brightness_params: Dict,
    ratio_mode: str = "softmin",
    delta_step: int = 2,
    hog_num_bins: int = 8,
    hog_sigma: float = 1.0,
) -> CenterScores:
    """
    For one patch center in image1:
        - build patch P once
        - build descriptors of P once
        - scan all candidate deltas
        - keep best color scores and best HOG score
    """
    P = build_rect_patch(image1, center1, patch_size)

    color_desc_P = build_patch_color_descriptor(
        patch_bgr=P,
        strong_params=strong_params,
        weak_params=weak_params,
        brightness_params=brightness_params,
    )
    hog_desc_P = compute_patch_hog_descriptor(
        patch_bgr=P,
        num_angle_bins=hog_num_bins,
        gaussian_sigma=hog_sigma,
    )

    pred_center2 = tuple((A @ np.array(center1, dtype=np.float32) + t).tolist())

    deltas = generate_deltas(radius=radius, step=delta_step)

    best_R_strong = None
    best_R_weak = None
    best_R_brightness = None
    best_d_hog = None

    best_delta_strong = None
    best_delta_weak = None
    best_delta_brightness = None
    best_delta_hog = None

    strong_active = len(color_desc_P.active_s) > 0
    weak_active = len(color_desc_P.active_w) > 0
    brightness_active = len(color_desc_P.active_b) > 0

    num_valid_candidates = 0

    for delta in deltas:
        if not candidate_patch_fully_inside_image(
            center1=center1,
            A=A,
            t=t,
            delta=delta,
            patch_size=patch_size,
            image_shape=image2.shape[:2],
        ):
            continue

        Q = build_canonical_candidate_patch(
            image2=image2,
            center1=center1,
            A=A,
            t=t,
            delta=delta,
            patch_size=patch_size,
        )

        color_desc_Q = build_patch_color_descriptor(
            patch_bgr=Q,
            strong_params=strong_params,
            weak_params=weak_params,
            brightness_params=brightness_params,
        )
        color_scores = compare_patch_color_descriptors(
            desc_ref=color_desc_P,
            desc_cmp=color_desc_Q,
            ratio_mode=ratio_mode,
            strong_softmin_p=strong_params.get("softmin_p", 8),
            weak_softmin_p=weak_params.get("softmin_p", 8),
            brightness_softmin_p=brightness_params.get("softmin_p", 8),
        )

        hog_desc_Q = compute_patch_hog_descriptor(
            patch_bgr=Q,
            num_angle_bins=hog_num_bins,
            gaussian_sigma=hog_sigma,
        )
        d_hog = compare_hog_descriptors(hog_desc_P, hog_desc_Q)

        num_valid_candidates += 1

        if color_scores.strong.active and color_scores.strong.score is not None:
            if best_R_strong is None or color_scores.strong.score > best_R_strong:
                best_R_strong = color_scores.strong.score
                best_delta_strong = delta

        if color_scores.weak.active and color_scores.weak.score is not None:
            if best_R_weak is None or color_scores.weak.score > best_R_weak:
                best_R_weak = color_scores.weak.score
                best_delta_weak = delta

        if color_scores.brightness.active and color_scores.brightness.score is not None:
            if best_R_brightness is None or color_scores.brightness.score > best_R_brightness:
                best_R_brightness = color_scores.brightness.score
                best_delta_brightness = delta

        if best_d_hog is None or d_hog < best_d_hog:
            best_d_hog = d_hog
            best_delta_hog = delta

    return CenterScores(
        center1=center1,
        pred_center2=pred_center2,

        best_R_strong=best_R_strong,
        best_R_weak=best_R_weak,
        best_R_brightness=best_R_brightness,
        best_d_hog=best_d_hog,

        best_delta_strong=best_delta_strong,
        best_delta_weak=best_delta_weak,
        best_delta_brightness=best_delta_brightness,
        best_delta_hog=best_delta_hog,

        strong_active=strong_active,
        weak_active=weak_active,
        brightness_active=brightness_active,

        num_valid_candidates=num_valid_candidates,
    )


# ============================================================
# DECISION RULES FOR ONE PATCH
# ============================================================

def decide_similarity_channel(
    score: Optional[float],
    active: bool,
    valid: bool,
    thr_lo: float,
    thr_hi: float,
    higher_is_better: bool,
) -> str:
    """
    Return:
        - "MATCH"
        - "MISMATCH"
        - "UNKNOWN"
    """
    if (not valid) or (not active) or (score is None):
        return "UNKNOWN"

    if higher_is_better:
        if score <= thr_lo:
            return "MISMATCH"
        elif score >= thr_hi:
            return "MATCH"
        else:
            return "UNKNOWN"
    else:
        if score <= thr_lo:
            return "MATCH"
        elif score >= thr_hi:
            return "MISMATCH"
        else:
            return "UNKNOWN"


def decide_center_scores(
    center_scores: CenterScores,
    T_s_lo: float = 0.45,
    T_s_hi: float = 0.75,
    T_w_lo: float = 0.35,
    T_w_hi: float = 0.70,
    T_b_lo: float = 0.50,
    T_b_hi: float = 0.80,
    D_h_lo: float = 0.12,
    D_h_hi: float = 0.25,
) -> Dict:
    """
    Independent decision for each channel.
    Patch is MISMATCH if at least one channel is MISMATCH.
    """
    valid_any = center_scores.num_valid_candidates > 0

    strong_state = decide_similarity_channel(
        score=center_scores.best_R_strong,
        active=center_scores.strong_active,
        valid=valid_any,
        thr_lo=T_s_lo,
        thr_hi=T_s_hi,
        higher_is_better=True,
    )

    weak_state = decide_similarity_channel(
        score=center_scores.best_R_weak,
        active=center_scores.weak_active,
        valid=valid_any,
        thr_lo=T_w_lo,
        thr_hi=T_w_hi,
        higher_is_better=True,
    )

    brightness_state = decide_similarity_channel(
        score=center_scores.best_R_brightness,
        active=center_scores.brightness_active,
        valid=valid_any,
        thr_lo=T_b_lo,
        thr_hi=T_b_hi,
        higher_is_better=True,
    )

    hog_state = decide_similarity_channel(
        score=center_scores.best_d_hog,
        active=True,
        valid=valid_any and (center_scores.best_d_hog is not None),
        thr_lo=D_h_lo,
        thr_hi=D_h_hi,
        higher_is_better=False,
    )

    states = [strong_state, weak_state, brightness_state, hog_state]

    if "MISMATCH" in states:
        patch_state = "PATCH_MISMATCH"
    elif "MATCH" in states:
        patch_state = "PATCH_MATCH"
    else:
        patch_state = "PATCH_UNKNOWN"

    return {
        "patch_state": patch_state,
        "strong_state": strong_state,
        "weak_state": weak_state,
        "brightness_state": brightness_state,
        "hog_state": hog_state,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def score_all_centers(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    patch_size: int = 48,
    step_diag_frac: float = 0.06,
    search_radius: Optional[int] = None,
    delta_step: int = 2,
    strong_params: Optional[Dict] = None,
    weak_params: Optional[Dict] = None,
    brightness_params: Optional[Dict] = None,
    ratio_mode: str = "softmin",
    hog_num_bins: int = 8,
    hog_sigma: float = 1.0,
    affine_ransac_thresh: float = 3.0,
) -> List[Dict]:
    """
    Full pipeline.

    Input:
        image1, image2: BGR images
        mask1, mask2: search masks
        pts1, pts2: matched keypoints, shape N x 2

    Output:
        list of dictionaries, one per patch center
    """
    if strong_params is None or weak_params is None or brightness_params is None:
        raise ValueError("strong_params, weak_params, brightness_params must be provided")

    A, t, inliers = estimate_global_affine(
        pts1=pts1,
        pts2=pts2,
        ransac_thresh=affine_ransac_thresh,
    )

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    if search_radius is None:
        search_radius = auto_search_radius_from_affine_errors(
            pts1=pts1_in,
            pts2=pts2_in,
            A=A,
            t=t,
        )

    centers = generate_patch_centers_from_mask_bbox(
        mask=mask1,
        patch_size=patch_size,
        step_diag_frac=step_diag_frac,
    )

    results = []
    for center1 in centers:
        center_scores = search_scores_for_center(
            image1=image1,
            image2=image2,
            center1=center1,
            A=A,
            t=t,
            patch_size=patch_size,
            radius=search_radius,
            strong_params=strong_params,
            weak_params=weak_params,
            brightness_params=brightness_params,
            ratio_mode=ratio_mode,
            delta_step=delta_step,
            hog_num_bins=hog_num_bins,
            hog_sigma=hog_sigma,
        )

        decision = decide_center_scores(center_scores)

        results.append({
            "center1": center_scores.center1,
            "pred_center2": center_scores.pred_center2,

            "best_R_strong": center_scores.best_R_strong,
            "best_R_weak": center_scores.best_R_weak,
            "best_R_brightness": center_scores.best_R_brightness,
            "best_d_hog": center_scores.best_d_hog,

            "best_delta_strong": center_scores.best_delta_strong,
            "best_delta_weak": center_scores.best_delta_weak,
            "best_delta_brightness": center_scores.best_delta_brightness,
            "best_delta_hog": center_scores.best_delta_hog,

            "strong_active": center_scores.strong_active,
            "weak_active": center_scores.weak_active,
            "brightness_active": center_scores.brightness_active,

            "num_valid_candidates": center_scores.num_valid_candidates,

            "patch_state": decision["patch_state"],
            "strong_state": decision["strong_state"],
            "weak_state": decision["weak_state"],
            "brightness_state": decision["brightness_state"],
            "hog_state": decision["hog_state"],

            "search_radius": search_radius,
            "num_inliers_affine": int(inliers.sum()),
        })

    return results
