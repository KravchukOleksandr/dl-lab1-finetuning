import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_mask_by_points(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Crop a binary mask by the tight bounding box of keypoints.

    Args:
        mask:
            Binary mask of shape (H, W).
        points:
            Array of shape (N, 2) with point coordinates in (x, y) format.

    Returns:
        Cropped mask of the same shape.
    """
    x = points[:, 0]
    y = points[:, 1]

    x1 = max(0, int(np.floor(x.min())))
    x2 = min(mask.shape[1], int(np.ceil(x.max())))
    y1 = max(0, int(np.floor(y.min())))
    y2 = min(mask.shape[0], int(np.ceil(y.max())))

    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def cut_top_half(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the lower half of a mask.

    Args:
        mask:
            Binary mask of shape (H, W).

    Returns:
        Mask of the same shape with the top half removed.
    """
    h = mask.shape[0]
    out = mask.copy()
    out[: h // 2, :] = 0
    return out


def compute_color_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    bins: int = 16,
) -> np.ndarray:
    """
    Compute a normalized 2D Lab (a,b) color histogram inside a mask.

    Args:
        image:
            BGR uint8 image.
        mask:
            Binary mask of shape (H, W).
        bins:
            Number of bins per channel.

    Returns:
        Histogram of shape (bins, bins).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1]
    b = lab[..., 2]

    valid = mask > 0

    hist, _, _ = np.histogram2d(
        a[valid],
        b[valid],
        bins=bins,
        range=[[0, 256], [0, 256]],
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compare two histograms with Bhattacharyya distance.

    Args:
        hist1:
            First histogram.
        hist2:
            Second histogram.

    Returns:
        Bhattacharyya distance.
    """
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))


def plot_histograms(hist1: np.ndarray, hist2: np.ndarray) -> None:
    """
    Plot two histograms side by side and show their distance.

    Args:
        hist1:
            First histogram.
        hist2:
            Second histogram.
    """
    dist = compare_histograms(hist1, hist2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(hist1, cmap="hot")
    axes[0].set_title("Hist 1")
    axes[0].axis("off")

    axes[1].imshow(hist2, cmap="hot")
    axes[1].set_title("Hist 2")
    axes[1].axis("off")

    plt.suptitle(f"Bhattacharyya distance: {dist:.4f}")
    plt.tight_layout()
    plt.show()



# 1. Tight crop of masks by matched keypoints
mask1_cropped = crop_mask_by_points(mask1, src_pts)
mask2_cropped = crop_mask_by_points(mask2, dst_pts)

# 2. Remove top half
mask1_final = cut_top_half(mask1_cropped)
mask2_final = cut_top_half(mask2_cropped)

# 3. Build histograms
hist1 = compute_color_histogram(img1, mask1_final, bins=16)
hist2 = compute_color_histogram(img2, mask2_final, bins=16)

# 4. Compare
dist = compare_histograms(hist1, hist2)
print("distance:", dist)

# 5. Visualize
plot_histograms(hist1, hist2)