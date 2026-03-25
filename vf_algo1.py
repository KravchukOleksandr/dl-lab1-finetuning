from __future__ import annotations

import cv2
import numpy as np


def points_to_density_map(
    points: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    sigma: float = 12.0,
) -> np.ndarray:
    """
    Convert 2D points into a smooth normalized density map.

    Each point is rendered into the map and the result is smoothed with a
    Gaussian blur. The final map is normalized to the range [0, 1].

    Args:
        points:
            Array of shape (N, 2) with point coordinates in (x, y) format.
        image_shape:
            Image shape as (H, W) or (H, W, C).
        sigma:
            Gaussian smoothing sigma in pixels.

    Returns:
        A float32 density map with shape (H, W) normalized to [0, 1].
    """
    h, w = image_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)

    np.add.at(density, (ys, xs), 1.0)

    density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)

    max_value = float(density.max())
    if max_value > 0.0:
        density /= max_value

    return density


def build_diff_map(
    self_points: np.ndarray,
    cross_points: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    sigma: float = 12.0,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build self, cross, and diff density maps.

    The diff map is defined as:
        relu(self_density - alpha * cross_density)

    Args:
        self_points:
            Stable self-match points in (x, y) format.
        cross_points:
            Cross-match points in the same image coordinate system.
        image_shape:
            Image shape as (H, W) or (H, W, C).
        sigma:
            Gaussian smoothing sigma in pixels.
        alpha:
            Cross-map weight inside the subtraction.

    Returns:
        A tuple:
            (self_density, cross_density, diff_map)
    """
    self_density = points_to_density_map(self_points, image_shape, sigma=sigma)
    cross_density = points_to_density_map(cross_points, image_shape, sigma=sigma)
    diff_map = np.maximum(self_density - alpha * cross_density, 0.0)

    max_value = float(diff_map.max())
    if max_value > 0.0:
        diff_map /= max_value

    return self_density, cross_density, diff_map


def extract_diff_regions(
    diff_map: np.ndarray,
    threshold: float = 0.2,
    min_area: int = 150,
    min_mass: float = 20.0,
) -> list[dict]:
    """
    Extract connected mismatch regions from a diff map.

    Args:
        diff_map:
            Diff map in the range [0, 1].
        threshold:
            Threshold used to binarize the diff map.
        min_area:
            Minimum connected-component area in pixels.
        min_mass:
            Minimum summed diff value inside a component.

    Returns:
        A list of region dictionaries sorted by descending mass.
        Each region contains:
            - label
            - area
            - mass
            - peak
            - bbox as (x, y, w, h)
            - centroid as (cx, cy)
            - mask
    """
    binary = (diff_map >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    regions: list[dict] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        region_mask = labels == label
        region_values = diff_map[region_mask]
        mass = float(region_values.sum())

        if area < min_area or mass < min_mass:
            continue

        regions.append(
            {
                "label": label,
                "area": int(area),
                "mass": mass,
                "peak": float(region_values.max()),
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (float(centroids[label][0]), float(centroids[label][1])),
                "mask": region_mask,
            }
        )

    regions.sort(key=lambda r: r["mass"], reverse=True)
    return regions


def overlay_density_map(
    image: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a density map on top of a BGR image.

    Args:
        image:
            Input BGR image.
        density_map:
            Float map in the range [0, 1].
        alpha:
            Heatmap blending strength.
        colormap:
            OpenCV colormap.

    Returns:
        A BGR image with the density map overlaid.
    """
    heat = np.clip(density_map * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, colormap)
    return cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0.0)