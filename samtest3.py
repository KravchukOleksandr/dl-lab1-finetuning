import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM


IMAGE_PATH = "frame.png"
LABEL_PATH = "frame.txt"

PERSON_CLASS_ID = 0

# Варианты:
# "sam2.1_l.pt" / "sam2.1_b.pt" / "sam2_l.pt" / "sam_b.pt"
# Название зависит от того, какие веса доступны/скачаны в твоей среде.
SAM_MODEL_PATH = "sam2.1_l.pt"

DEVICE = 0  # 0 для cuda:0, "cpu" для CPU

EXPAND_SMALL = 2.5
EXPAND_MEDIUM = 1.8
EXPAND_LARGE = 1.25

ALPHA = 0.45


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
        scale = EXPAND_SMALL
    elif bw < 60 or bh < 140:
        scale = EXPAND_MEDIUM
    else:
        scale = EXPAND_LARGE

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    expanded = np.array([
        cx - bw * scale / 2,
        cy - bh * scale / 2,
        cx + bw * scale / 2,
        cy + bh * scale / 2,
    ], dtype=np.float32)

    return clip_box(expanded, img_w, img_h)


def overlay_mask(image_rgb, mask, box=None):
    overlay = image_rgb.copy().astype(np.float32)

    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask > 0] = (1 - ALPHA) * overlay[mask > 0] + ALPHA * red
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
boxes_original = []
boxes_prompt = []

with open(LABEL_PATH, "r") as f:
    for line in f:
        row = list(map(float, line.strip().split()))

        if len(row) != 5:
            continue

        if int(row[0]) != PERSON_CLASS_ID:
            continue

        original_box = yolo_to_xyxy(row, img_w, img_h)
        original_box = clip_box(original_box, img_w, img_h)

        prompt_box = expand_box(original_box, img_w, img_h)

        boxes_original.append(original_box)
        boxes_prompt.append(prompt_box)

# Run Ultralytics SAM/SAM2
model = SAM(SAM_MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    bboxes=[box.tolist() for box in boxes_prompt],
    device=DEVICE,
    retina_masks=True,
    verbose=False,
)

result = results[0]

if result.masks is None:
    print("No masks returned")
else:
    masks = result.masks.data.cpu().numpy()

    n = len(masks)

    plt.figure(figsize=(15, 5 * n))

    for i in range(n):
        mask = masks[i].astype(bool)
        original_box = boxes_original[i]

        image_with_box = image_rgb.copy()
        x1, y1, x2, y2 = original_box.astype(int)
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

        overlay = overlay_mask(image_rgb, mask, original_box)

        ax = plt.subplot(n, 3, i * 3 + 1)
        ax.imshow(image_with_box)
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