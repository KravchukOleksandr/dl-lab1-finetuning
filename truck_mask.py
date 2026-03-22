from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_truck_candidates(result, img_w, img_h):
    names = result.names
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    clses = result.boxes.cls.cpu().numpy().astype(int)

    candidates = []
    for i, cls_id in enumerate(clses):
        if names[int(cls_id)] != "truck":
            continue

        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        area = w * h

        mask = (masks[i] > 0.5).astype(np.uint8) * 255
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        candidates.append({
            "box": (x1, y1, x2, y2),
            "w": w,
            "h": h,
            "area": area,
            "mask": mask,
        })

    return candidates


def save_crop_and_mask(img, obj, crop_path, mask_path):
    x1, y1, x2, y2 = obj["box"]
    crop = img[y1:y2, x1:x2]
    mask_crop = obj["mask"][y1:y2, x1:x2]

    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(mask_path), mask_crop)


def run_case1(in_dir, out_dir, model_path="yolov8n-seg.pt"):
    """
    Ограничение:
    aspect ratio кропа <= 2 и >= 0.5
    """
    model = YOLO(model_path)

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    masks_dir = out_dir / "masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(in_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        result = model.predict(source=img, verbose=False)[0]
        candidates = get_truck_candidates(result, w, h)

        candidates = [
            c for c in candidates
            if 0.5 <= (c["w"] / c["h"]) <= 2.0
        ]

        best = max(candidates, key=lambda c: c["area"])
        save_crop_and_mask(
            img, best,
            crops_dir / img_path.name,
            masks_dir / img_path.name
        )


def run_case2(in_dir, out_dir, model_path="yolov8n-seg.pt"):
    """
    Ограничение:
    кроп не должен касаться левой четверти кадра
    => x1 >= W / 4
    """
    model = YOLO(model_path)

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    masks_dir = out_dir / "masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(in_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        result = model.predict(source=img, verbose=False)[0]
        candidates = get_truck_candidates(result, w, h)

        candidates = [
            c for c in candidates
            if c["box"][0] >= w / 4
        ]

        best = max(candidates, key=lambda c: c["area"])
        save_crop_and_mask(
            img, best,
            crops_dir / img_path.name,
            masks_dir / img_path.name
        )


if __name__ == "__main__":
    # 1-й случай
    run_case1("input_images", "output_case1")

    # 2-й случай
    run_case2("input_images", "output_case2")
