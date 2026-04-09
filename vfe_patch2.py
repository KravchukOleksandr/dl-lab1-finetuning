import math
import cv2
import numpy as np


# ============================================================
# Expected external functions
# ============================================================
# compute_soft_color_histogram(
#     image, mask,
#     num_bins, sigma_bins,
#     low_chroma_center, low_chroma_sharpness,
#     high_chroma_center, high_chroma_sharpness
# ) -> hist
#
# compute_soft_brightness_histogram(
#     image, mask,
#     num_bins, sigma_bins,
#     low_chroma_center, low_chroma_sharpness,
#     high_chroma_center, high_chroma_sharpness
# ) -> hist
# ============================================================


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def estimate_global_affine(pts1, pts2, ransac_thresh=3.0):
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
    return A, t, inliers.ravel().astype(bool)


def affine_transfer_errors(pts1, pts2, A, t):
    pred = pts1 @ A.T + t[None, :]
    return np.linalg.norm(pred - pts2, axis=1)


def auto_search_radius_from_affine_errors(
    pts1, pts2, A, t,
    q=0.90, scale=2.0,
    r_min=8, r_max=30
):
    err = affine_transfer_errors(pts1, pts2, A, t)
    e_q = float(np.quantile(err, q))
    r = int(round(scale * e_q))
    return max(r_min, min(r_max, r))


def build_canonical_patch(image, center, patch_size):
    S = patch_size
    half = (S - 1) / 2.0

    u = np.arange(S, dtype=np.float32) - half
    v = np.arange(S, dtype=np.float32) - half
    uu, vv = np.meshgrid(u, v)

    x = center[0] + uu
    y = center[1] + vv

    patch = cv2.remap(
        image,
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return patch


def patch_mask_occupancy(mask, center, patch_size):
    patch = build_canonical_patch(mask.astype(np.float32), center, patch_size)
    return float((patch > 0.5).mean())


def generate_positions_1d(start, end, step):
    if start > end:
        return []
    vals = list(range(start, end + 1, step))
    if len(vals) == 0:
        return []
    if vals[-1] != end:
        vals.append(end)
    return vals


def generate_patch_centers(mask, patch_size=48, step_diag_frac=0.06, min_occupancy=0.8):
    H, W = mask.shape[:2]
    x, y, w, h = bbox_from_mask(mask)

    diag = math.sqrt(w * w + h * h)
    step = max(1, int(round(step_diag_frac * diag)))

    half = (patch_size - 1) / 2.0

    x_start = int(math.ceil(max(x + half, half)))
    x_end = int(math.floor(min(x + w - 1 - half, W - 1 - half)))
    y_start = int(math.ceil(max(y + half, half)))
    y_end = int(math.floor(min(y + h - 1 - half, H - 1 - half)))

    xs = generate_positions_1d(x_start, x_end, step)
    ys = generate_positions_1d(y_start, y_end, step)

    centers = []
    for cy in ys:
        for cx in xs:
            occ = patch_mask_occupancy(mask, (cx, cy), patch_size)
            if occ >= min_occupancy:
                centers.append((cx, cy))

    return centers


def candidate_patch_fully_inside_image(center1, A, t, delta, patch_size, image_shape):
    H, W = image_shape[:2]
    half = (patch_size - 1) / 2.0

    corners_local = np.array([
        [-half, -half],
        [ half, -half],
        [-half,  half],
        [ half,  half],
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


def build_canonical_candidate_patch(image2, center1, A, t, delta, patch_size):
    S = patch_size
    half = (S - 1) / 2.0

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


def generate_deltas(radius, step=2):
    deltas = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx * dx + dy * dy <= radius * radius:
                deltas.append((dx, dy))
    return deltas


def windowize_circular(hist):
    n = len(hist)
    W = np.zeros_like(hist, dtype=np.float32)
    for i in range(n):
        W[i] = hist[(i - 1) % n] + hist[i] + hist[(i + 1) % n]
    return W


def windowize_linear(hist):
    n = len(hist)
    W = np.zeros_like(hist, dtype=np.float32)
    for i in range(n):
        W[i] = hist[max(i - 1, 0)] + hist[i] + hist[min(i + 1, n - 1)]
    return W


def get_active_indices(W_ref, min_fraction):
    return [i for i in range(len(W_ref)) if float(W_ref[i]) >= float(min_fraction)]


def directed_ratios(W_ref, W_cmp, active_indices, eps=1e-8):
    ratios = []
    for i in active_indices:
        r = float(W_cmp[i]) / (float(W_ref[i]) + eps)
        ratios.append(min(1.0, r))
    return ratios


def aggregate_ratios(ratios, mode="softmin", softmin_p=8, eps=1e-8):
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


def build_patch_color_descriptor(patch_bgr, strong_params, weak_params, brightness_params):
    S = patch_bgr.shape[0]
    patch_mask = np.ones((S, S), dtype=np.uint8)

    hist_s = compute_soft_color_histogram(
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

    hist_w = compute_soft_color_histogram(
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

    hist_b = compute_soft_brightness_histogram(
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

    return {
        "W_s": W_s, "active_s": active_s,
        "W_w": W_w, "active_w": active_w,
        "W_b": W_b, "active_b": active_b,
    }


def compare_patch_color_descriptors(
    desc_ref, desc_cmp,
    ratio_mode="softmin",
    strong_softmin_p=8,
    weak_softmin_p=8,
    brightness_softmin_p=8,
):
    ratios_s = directed_ratios(desc_ref["W_s"], desc_cmp["W_s"], desc_ref["active_s"])
    ratios_w = directed_ratios(desc_ref["W_w"], desc_cmp["W_w"], desc_ref["active_w"])
    ratios_b = directed_ratios(desc_ref["W_b"], desc_cmp["W_b"], desc_ref["active_b"])

    return {
        "R_strong": aggregate_ratios(ratios_s, ratio_mode, strong_softmin_p),
        "R_weak": aggregate_ratios(ratios_w, ratio_mode, weak_softmin_p),
        "R_brightness": aggregate_ratios(ratios_b, ratio_mode, brightness_softmin_p),
    }


def extract_L_channel(patch_bgr):
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 0] * (100.0 / 255.0)


def add_soft_orientation_vote(hist, theta, mag):
    n = len(hist)
    bin_pos = (theta / math.pi) * n
    i0 = int(math.floor(bin_pos)) % n
    frac = bin_pos - math.floor(bin_pos)
    i1 = (i0 + 1) % n
    hist[i0] += mag * (1.0 - frac)
    hist[i1] += mag * frac


def compute_patch_hog_descriptor(patch_bgr, num_angle_bins=8, gaussian_sigma=1.0):
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

    parts = []

    h_global = np.zeros(num_angle_bins, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            add_soft_orientation_vote(h_global, float(ang[y, x]), float(mag[y, x]))
    h_global /= (np.linalg.norm(h_global) + 1e-8)
    parts.append(h_global)

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
        parts.append(h_cell)

    desc = np.concatenate(parts, axis=0)
    desc /= (np.linalg.norm(desc) + 1e-8)
    return desc.astype(np.float32)


def compare_hog_descriptors(desc1, desc2):
    return float(1.0 - np.dot(desc1, desc2))


def score_all_centers(
    image1,
    mask1,
    image2,
    mask2,   # kept in signature for consistency, not used
    pts1,
    pts2,
    patch_size=48,
    step_diag_frac=0.06,
    min_center_occupancy=0.8,
    search_radius=None,
    delta_step=2,
    strong_params=None,
    weak_params=None,
    brightness_params=None,
    ratio_mode="softmin",
    hog_num_bins=8,
    hog_sigma=1.0,
    affine_ransac_thresh=3.0,
):
    """
    One-way scoring:
        patches are placed on image1
        best candidates are searched on image2

    Output:
        list of dicts with only:
            center1
            best_R_strong
            best_R_weak
            best_R_brightness
            best_d_hog
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

    centers = generate_patch_centers(
        mask=mask1,
        patch_size=patch_size,
        step_diag_frac=step_diag_frac,
        min_occupancy=min_center_occupancy,
    )

    deltas = generate_deltas(search_radius, step=delta_step)

    results = []

    for center1 in centers:
        P = build_canonical_patch(image1, center1, patch_size)

        color_desc_P = build_patch_color_descriptor(
            P, strong_params, weak_params, brightness_params
        )
        hog_desc_P = compute_patch_hog_descriptor(
            P, num_angle_bins=hog_num_bins, gaussian_sigma=hog_sigma
        )

        best_R_strong = None
        best_R_weak = None
        best_R_brightness = None
        best_d_hog = None

        for delta in deltas:
            if not candidate_patch_fully_inside_image(
                center1, A, t, delta, patch_size, image2.shape[:2]
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
                Q, strong_params, weak_params, brightness_params
            )
            color_scores = compare_patch_color_descriptors(
                color_desc_P,
                color_desc_Q,
                ratio_mode=ratio_mode,
                strong_softmin_p=strong_params.get("softmin_p", 8),
                weak_softmin_p=weak_params.get("softmin_p", 8),
                brightness_softmin_p=brightness_params.get("softmin_p", 8),
            )

            hog_desc_Q = compute_patch_hog_descriptor(
                Q, num_angle_bins=hog_num_bins, gaussian_sigma=hog_sigma
            )
            d_hog = compare_hog_descriptors(hog_desc_P, hog_desc_Q)

            if color_scores["R_strong"] is not None:
                if best_R_strong is None or color_scores["R_strong"] > best_R_strong:
                    best_R_strong = color_scores["R_strong"]

            if color_scores["R_weak"] is not None:
                if best_R_weak is None or color_scores["R_weak"] > best_R_weak:
                    best_R_weak = color_scores["R_weak"]

            if color_scores["R_brightness"] is not None:
                if best_R_brightness is None or color_scores["R_brightness"] > best_R_brightness:
                    best_R_brightness = color_scores["R_brightness"]

            if best_d_hog is None or d_hog < best_d_hog:
                best_d_hog = d_hog

        results.append({
            "center1": center1,
            "best_R_strong": best_R_strong,
            "best_R_weak": best_R_weak,
            "best_R_brightness": best_R_brightness,
            "best_d_hog": best_d_hog,
        })

    return results
