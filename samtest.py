import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


CROP_DIR = "out/crops"
MASK_DIR = "out/masks"

MAX_SHOW = 20
ALPHA = 0.45


crop_files = sorted([
    f for f in os.listdir(CROP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

crop_files = crop_files[:MAX_SHOW]

n = len(crop_files)

plt.figure(figsize=(10, 3 * n))

for i, filename in enumerate(crop_files):
    crop_path = os.path.join(CROP_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    crop_bgr = cv2.imread(crop_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    mask_float = (mask > 127).astype(np.float32)

    overlay = crop_rgb.copy().astype(np.float32)
    color = np.array([255, 0, 0], dtype=np.float32)

    overlay[mask_float > 0] = (
        (1 - ALPHA) * overlay[mask_float > 0]
        + ALPHA * color
    )

    overlay = overlay.astype(np.uint8)

    # Original
    ax = plt.subplot(n, 3, i * 3 + 1)
    ax.imshow(crop_rgb)
    ax.set_title("Original")
    ax.axis("off")

    # Mask
    ax = plt.subplot(n, 3, i * 3 + 2)
    ax.imshow(mask, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    # Overlay
    ax = plt.subplot(n, 3, i * 3 + 3)
    ax.imshow(overlay)
    ax.set_title("Overlay")
    ax.axis("off")

plt.tight_layout()
plt.show()