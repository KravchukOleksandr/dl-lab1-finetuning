import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_PATH = "frame.png"
LABEL_PATH = "frame.txt"

PERSON_CLASS_ID = 0

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def yolo_to_xyxy(row, img_w, img_h):
    cls, cx, cy, w, h = row

    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    return np.array([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ], dtype=np.float32)


def clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box

    return np.array([
        np.clip(x1, 0, img_w - 1),
        np.clip(y1, 0, img_h - 1),
        np.clip(x2, 0, img_w - 1),
        np.clip(y2, 0, img_h - 1),
    ], dtype=np.float32)


def expand_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < 25 or bh < 60:
        scale = 2.5
    elif bw < 60 or bh < 140:
        scale = 1.8
    else:
        scale = 1.25

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    expanded = np.array([
        cx - bw * scale / 2,
        cy - bh * scale / 2,
        cx + bw * scale / 2,
        cy + bh * scale / 2,
    ], dtype=np.float32)

    return clip_box(expanded, img_w, img_h)


def show_mask_overlay(image_rgb, mask, box=None, alpha=0.45):
    overlay = image_rgb.copy().astype(np.float32)

    color = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * color
    overlay = overlay.astype(np.uint8)

    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return overlay


# Load image
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
img_h, img_w = image_rgb.shape[:2]

# Load YOLO labels
labels = []
with open(LABEL_PATH, "r") as f:
    for line in f:
        row = list(map(float, line.strip().split()))
        if len(row) == 5 and int(row[0]) == PERSON_CLASS_ID:
            labels.append(row)

# Load SAM2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# Predict masks
with torch.inference_mode():
    if DEVICE.startswith("cuda"):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.no_grad()

    with ctx:
        predictor.set_image(image_rgb)

        rows = []

        for row in labels:
            original_box = yolo_to_xyxy(row, img_w, img_h)
            original_box = clip_box(original_box, img_w, img_h)
            prompt_box = expand_box(original_box, img_w, img_h)

            masks, scores, logits = predictor.predict(
                box=prompt_box,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(bool)

            overlay = show_mask_overlay(image_rgb, mask, original_box)

            rows.append((original_box, mask, overlay))

# Show
n = len(rows)

plt.figure(figsize=(15, 5 * n))

for i, (box, mask, overlay) in enumerate(rows):
    ax = plt.subplot(n, 3, i * 3 + 1)
    img_with_box = image_rgb.copy()
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ax.imshow(img_with_box)
    ax.set_title(f"Original #{i}")
    ax.axis("off")

    ax = plt.subplot(n, 3, i * 3 + 2)
    ax.imshow(mask, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = plt.subplot(n, 3, i * 3 + 3)
    ax.imshow(overlay)
    ax.set_title("Overlay")
    ax.axis("off")

plt.tight_layout()
plt.show()