import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# 1. Crop mask by keypoints
# -------------------------
def crop_mask_by_points(mask: np.ndarray, points: np.ndarray, margin: int = 5) -> np.ndarray:
    """
    Zero-out everything outside bbox of keypoints.
    """
    x = points[:, 0]
    y = points[:, 1]

    x1 = max(0, int(x.min()) - margin)
    x2 = min(mask.shape[1], int(x.max()) + margin)

    y1 = max(0, int(y.min()) - margin)
    y2 = min(mask.shape[0], int(y.max()) + margin)

    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


# -------------------------
# 2. Paint mask (adaptive)
# -------------------------
def compute_paint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    quantile: float = 0.4,
    cut_top: float = 0.2,
) -> np.ndarray:
    """
    Extract smooth (paint-like) regions using gradient quantile.

    - keeps lowest-gradient pixels inside mask
    - optionally removes top part (windshield heuristic)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)

    valid = mask > 0
    thr = np.quantile(grad[valid], quantile)

    paint = valid & (grad <= thr)

    # cut top region (windshield suppression)
    if cut_top > 0:
        h = image.shape[0]
        paint[: int(h * cut_top), :] = False

    return paint


# -------------------------
# 3. Histogram (a,b)
# -------------------------
def compute_color_histogram(
    image: np.ndarray,
    paint_mask: np.ndarray,
    bins: int = 16,
    min_bin: float = 0.01,
) -> np.ndarray:
    """
    Compute robust color histogram (Lab a,b) with small-bin suppression.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1]
    b = lab[..., 2]

    a_vals = a[paint_mask]
    b_vals = b[paint_mask]

    hist, _, _ = np.histogram2d(
        a_vals,
        b_vals,
        bins=bins,
        range=[[0, 256], [0, 256]],
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6

    # remove weak colors
    hist[hist < min_bin] = 0
    hist /= hist.sum() + 1e-6

    return hist


# -------------------------
# 4. Distance
# -------------------------
def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Bhattacharyya distance (0 = same, 1 = different)
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


# -------------------------
# 5. Visualization
# -------------------------
def plot_histograms(hist1: np.ndarray, hist2: np.ndarray):
    """
    Show two histograms side by side with distance.
    """
    dist = compare_histograms(hist1, hist2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(hist1, cmap="hot")
    axes[0].set_title("Hist 1")
    axes[0].axis("off")

    axes[1].imshow(hist2, cmap="hot")
    axes[1].set_title("Hist 2")
    axes[1].axis("off")

    plt.suptitle(f"Bhattacharyya distance: {dist:.3f}")
    plt.tight_layout()
    plt.show()

# crop
mask1_c = crop_mask_by_points(mask1, pts1)
mask2_c = crop_mask_by_points(mask2, pts2)

# paint
paint1 = compute_paint_mask(img1, mask1_c)
paint2 = compute_paint_mask(img2, mask2_c)

# hist
h1 = compute_color_histogram(img1, paint1)
h2 = compute_color_histogram(img2, paint2)

# compare
dist = compare_histograms(h1, h2)
print(dist)

# visualize
plot_histograms(h1, h2)