import cv2
import numpy as np


def segment_color_regions(image: np.ndarray, mask: np.ndarray):
    """
    Segment image into connected color regions inside mask.

    Args:
        image: BGR uint8 image
        mask: binary mask (same H,W)

    Returns:
        List of regions, each as dict:
            {
                "mask": np.ndarray (bool),
                "area": int,
                "centroid": (x, y),
                "color_ab": np.ndarray shape (2,)
            }
    """

    # --- Lab ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # --- Blur (важно) ---
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)

    # --- Quantization ---
    bin_size = 16  # можно 8 или 16
    a_q = (a // bin_size).astype(np.int32)
    b_q = (b // bin_size).astype(np.int32)

    label_map = a_q * 32 + b_q  # 32 = max bins по b

    # --- работаем только внутри mask ---
    valid = mask.astype(bool)

    regions = []

    unique_labels = np.unique(label_map[valid])

    for label in unique_labels:
        region_mask = (label_map == label) & valid

        # --- connected components ---
        num_labels, cc = cv2.connectedComponents(region_mask.astype(np.uint8))

        for i in range(1, num_labels):
            comp = (cc == i)

            area = comp.sum()
            if area < 200:  # фильтр шума
                continue

            ys, xs = np.where(comp)

            cx = xs.mean()
            cy = ys.mean()

            # median цвет
            color = np.median(
                np.stack([a[comp], b[comp]], axis=1),
                axis=0
            )

            regions.append({
                "mask": comp,
                "area": area,
                "centroid": (cx, cy),
                "color_ab": color,
            })

    return regions


from __future__ import annotations

import cv2
import numpy as np


def visualize_color_regions(
    image: np.ndarray,
    regions: list[dict],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Visualize segmented color regions.

    The output image contains:
    - left: region mask visualization
    - right: original image with region overlays

    Args:
        image:
            Original BGR image.
        regions:
            List of region descriptors returned by `segment_color_regions`.
        mask:
            Optional binary object mask. If provided, pixels outside the mask are
            darkened in the left panel.

    Returns:
        A BGR visualization image.
    """
    h, w = image.shape[:2]

    left = np.zeros((h, w, 3), dtype=np.uint8)
    right = image.copy()

    if mask is not None:
        left_mask = (mask > 0).astype(np.uint8)
    else:
        left_mask = np.ones((h, w), dtype=np.uint8)

    rng = np.random.default_rng(42)

    for region in regions:
        region_mask = region["mask"]

        color = rng.integers(64, 256, size=3, dtype=np.uint8).tolist()
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        left[region_mask] = color_tuple

        overlay = right.copy()
        overlay[region_mask] = color_tuple
        right = cv2.addWeighted(right, 0.75, overlay, 0.25, 0.0)

        ys, xs = np.where(region_mask)
        if len(xs) == 0:
            continue

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))

        cv2.rectangle(right, (x1, y1), (x2, y2), color_tuple, 1)
        cv2.circle(right, (cx, cy), 2, color_tuple, -1)

    if mask is not None:
        left[~left_mask.astype(bool)] = (20, 20, 20)

    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = left
    canvas[:, w:] = right

    cv2.putText(
        canvas,
        "regions",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "image",
        (w + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


regions = segment_color_regions(image, mask)
vis = visualize_color_regions(image, regions, mask)
cv2.imwrite("regions_debug.jpg", vis)