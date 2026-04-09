import cv2
import numpy as np
import math


def draw_polygon(image, pts, color, thickness=2):
    """
    Draw a closed polygon on image.
    pts: 4x2 float or int array
    """
    out = image.copy()
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts_i], isClosed=True, color=color, thickness=thickness)
    return out


def get_patch_corners(center, patch_size):
    """
    Return canonical square patch corners in image coordinates.
    """
    half = (patch_size - 1) / 2.0
    cx, cy = float(center[0]), float(center[1])

    corners = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)
    return corners


def transfer_patch_corners(center1, A, t, delta, patch_size):
    """
    Transfer patch corners from image1 to image2 using affine + delta.
    """
    corners1 = get_patch_corners(center1, patch_size)
    delta_vec = np.array(delta, dtype=np.float32)
    corners2 = corners1 @ A.T + t[None, :] + delta_vec[None, :]
    return corners2


def debug_single_center(
    image1,
    image2,
    center1,
    A,
    t,
    patch_size,
    radius,
    strong_params,
    weak_params,
    brightness_params,
    ratio_mode="softmin",
    delta_step=2,
    hog_num_bins=8,
    hog_sigma=1.0,
):
    """
    Debug one center only.

    Returns:
        {
            "debug_image1": ...,
            "debug_image2": ...,
            "metrics": {...}
        }
    """

    # Build source patch and descriptors
    P = build_canonical_patch(image1, center1, patch_size)

    color_desc_P = build_patch_color_descriptor(
        P, strong_params, weak_params, brightness_params
    )
    hog_desc_P = compute_patch_hog_descriptor(
        P, num_angle_bins=hog_num_bins, gaussian_sigma=hog_sigma
    )

    deltas = generate_deltas(radius, step=delta_step)

    best_R_strong = None
    best_R_weak = None
    best_R_brightness = None
    best_d_hog = None

    best_delta_strong = None
    best_delta_weak = None
    best_delta_brightness = None
    best_delta_hog = None

    num_valid_candidates = 0

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

        num_valid_candidates += 1

        if color_scores["R_strong"] is not None:
            if best_R_strong is None or color_scores["R_strong"] > best_R_strong:
                best_R_strong = color_scores["R_strong"]
                best_delta_strong = delta

        if color_scores["R_weak"] is not None:
            if best_R_weak is None or color_scores["R_weak"] > best_R_weak:
                best_R_weak = color_scores["R_weak"]
                best_delta_weak = delta

        if color_scores["R_brightness"] is not None:
            if best_R_brightness is None or color_scores["R_brightness"] > best_R_brightness:
                best_R_brightness = color_scores["R_brightness"]
                best_delta_brightness = delta

        if best_d_hog is None or d_hog < best_d_hog:
            best_d_hog = d_hog
            best_delta_hog = delta

    # Draw source patch on image1
    src_corners = get_patch_corners(center1, patch_size)
    debug_image1 = image1.copy()
    debug_image1 = draw_polygon(debug_image1, src_corners, color=(0, 255, 0), thickness=2)

    # Draw best patches on image2
    debug_image2 = image2.copy()

    if best_delta_strong is not None:
        poly = transfer_patch_corners(center1, A, t, best_delta_strong, patch_size)
        debug_image2 = draw_polygon(debug_image2, poly, color=(0, 255, 255), thickness=2)  # yellow

    if best_delta_weak is not None:
        poly = transfer_patch_corners(center1, A, t, best_delta_weak, patch_size)
        debug_image2 = draw_polygon(debug_image2, poly, color=(255, 0, 255), thickness=2)  # magenta

    if best_delta_brightness is not None:
        poly = transfer_patch_corners(center1, A, t, best_delta_brightness, patch_size)
        debug_image2 = draw_polygon(debug_image2, poly, color=(255, 255, 0), thickness=2)  # cyan

    if best_delta_hog is not None:
        poly = transfer_patch_corners(center1, A, t, best_delta_hog, patch_size)
        debug_image2 = draw_polygon(debug_image2, poly, color=(0, 0, 255), thickness=2)  # red

    metrics = {
        "center1": center1,
        "best_R_strong": best_R_strong,
        "best_R_weak": best_R_weak,
        "best_R_brightness": best_R_brightness,
        "best_d_hog": best_d_hog,
        "best_delta_strong": best_delta_strong,
        "best_delta_weak": best_delta_weak,
        "best_delta_brightness": best_delta_brightness,
        "best_delta_hog": best_delta_hog,
        "num_valid_candidates": num_valid_candidates,
    }

    return {
        "debug_image1": debug_image1,
        "debug_image2": debug_image2,
        "metrics": metrics,
    }
