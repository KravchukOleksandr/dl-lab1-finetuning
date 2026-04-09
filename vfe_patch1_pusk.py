import cv2
import numpy as np

# Example inputs
image1 = cv2.imread("truck1.jpg")
image2 = cv2.imread("truck2.jpg")

mask1 = cv2.imread("mask1.png", cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread("mask2.png", cv2.IMREAD_GRAYSCALE)

# Matched keypoints inside masks, shape N x 2
pts1 = np.array([
    [120.0, 220.0],
    [180.0, 250.0],
    [260.0, 240.0],
    [300.0, 180.0],
    [360.0, 210.0],
], dtype=np.float32)

pts2 = np.array([
    [132.0, 228.0],
    [191.0, 259.0],
    [274.0, 248.0],
    [312.0, 189.0],
    [372.0, 219.0],
], dtype=np.float32)

# Strong color parameters
strong_params = {
    "num_bins": 8,
    "sigma_bins": 1.5,
    "low_chroma_center": None,
    "low_chroma_sharpness": None,
    "high_chroma_center": 15.0,
    "high_chroma_sharpness": 3.0,
    "min_fraction": 0.04,
    "softmin_p": 8,
}

# Weak color parameters
weak_params = {
    "num_bins": 4,
    "sigma_bins": 1.25,
    "low_chroma_center": 5.0,
    "low_chroma_sharpness": 0.8,
    "high_chroma_center": 15.0,
    "high_chroma_sharpness": 3.0,
    "min_fraction": 0.06,
    "softmin_p": 8,
}

# Brightness parameters for low-chroma pixels
brightness_params = {
    "num_bins": 4,
    "sigma_bins": 1.0,
    "low_chroma_center": None,
    "low_chroma_sharpness": None,
    "high_chroma_center": 5.0,
    "high_chroma_sharpness": 0.8,
    "min_fraction": 0.10,
    "softmin_p": 8,
}

results = score_all_centers(
    image1=image1,
    mask1=mask1,
    image2=image2,
    mask2=mask2,
    pts1=pts1,
    pts2=pts2,
    patch_size=48,
    step_diag_frac=0.06,
    search_radius=None,      # auto from affine residuals
    delta_step=2,
    strong_params=strong_params,
    weak_params=weak_params,
    brightness_params=brightness_params,
    ratio_mode="softmin",    # or "min"
    hog_num_bins=8,
    hog_sigma=1.0,
    affine_ransac_thresh=3.0,
)

# Print a few results
for item in results[:10]:
    print(
        item["center1"],
        item["best_R_strong"],
        item["best_R_weak"],
        item["best_R_brightness"],
        item["best_d_hog"],
        item["patch_state"],
    )
