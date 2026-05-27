import io
import os
import json
import posixpath
from typing import List, Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import SAM
from azure.storage.blob import BlobServiceClient, ContentSettings


# =========================
# CONFIG
# =========================

AZURE_CONNECTION_STRING = "PUT_CONNECTION_STRING_HERE"

INPUT_CONTAINERS = [
    "container-1",
    "container-2",
]

OUTPUT_CONTAINER = "sam-crops-masks"

SAM_MODEL_PATH = "sam2.1_l.pt"  # если у тебя файл называется иначе, поставь свой путь

PERSON_CLASS_ID = 0

BOX_SCALE = 1.25

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = 0        # 0 = cuda:0, "cpu" = CPU
RETINA_MASKS = True

OVERWRITE = False

SAVE_META_JSON = True

CROP_SUFFIX = "_crop.png"
MASK_SUFFIX = "_mask.png"
META_SUFFIX = "_meta.json"


# =========================
# BASIC UTILS
# =========================

def is_image_blob(blob_name: str) -> bool:
    ext = os.path.splitext(blob_name.lower())[1]
    return ext in IMAGE_EXTS


def label_name_for_image(image_name: str) -> str:
    root, _ = os.path.splitext(image_name)
    return root + ".txt"


def yolo_to_xyxy(row: List[float], img_w: int, img_h: int) -> np.ndarray:
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


def clip_box_xyxy(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box

    return np.array([
        np.clip(x1, 0, img_w - 1),
        np.clip(y1, 0, img_h - 1),
        np.clip(x2, 0, img_w - 1),
        np.clip(y2, 0, img_h - 1),
    ], dtype=np.float32)


def expand_box_xyxy(box: np.ndarray, img_w: int, img_h: int, scale: float = BOX_SCALE) -> np.ndarray:
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_w = bw * scale
    new_h = bh * scale

    expanded = np.array([
        cx - new_w / 2.0,
        cy - new_h / 2.0,
        cx + new_w / 2.0,
        cy + new_h / 2.0,
    ], dtype=np.float32)

    return clip_box_xyxy(expanded, img_w, img_h)


def read_yolo_txt(txt: str) -> List[List[float]]:
    rows = []

    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        values = list(map(float, parts))
        rows.append(values)

    return rows


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Cannot decode image")

    return image_bgr


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)

    if not ok:
        raise ValueError("Cannot encode PNG")

    return buf.tobytes()


def crop_by_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box, w, h).astype(int)

    return image[y1:y2, x1:x2].copy()


def mask_to_crop(mask: np.ndarray, box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = clip_box_xyxy(box, img_w, img_h).astype(int)
    crop_mask = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    return crop_mask


def keep_mask_inside_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box, w, h).astype(int)

    out = np.zeros_like(mask, dtype=bool)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    return out


def safe_blob_prefix(source_container: str, image_blob_name: str) -> str:
    root, _ = os.path.splitext(image_blob_name)
    return posixpath.join(source_container, root).replace("\\", "/")


# =========================
# AZURE UTILS
# =========================

def download_blob_bytes(container_client, blob_name: str) -> bytes:
    return container_client.get_blob_client(blob_name).download_blob().readall()


def upload_bytes(container_client, blob_name: str, data: bytes, content_type: str):
    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=OVERWRITE,
        content_settings=ContentSettings(content_type=content_type),
    )


def blob_exists(container_client, blob_name: str) -> bool:
    try:
        container_client.get_blob_client(blob_name).get_blob_properties()
        return True
    except Exception:
        return False


# =========================
# MAIN PROCESSING
# =========================

def process_one_image(
    sam_model: SAM,
    input_container_name: str,
    input_container,
    output_container,
    image_blob_name: str,
    label_blob_name: str,
):
    image_bytes = download_blob_bytes(input_container, image_blob_name)
    label_bytes = download_blob_bytes(input_container, label_blob_name)

    image_bgr = decode_image(image_bytes)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    img_h, img_w = image_bgr.shape[:2]

    label_txt = label_bytes.decode("utf-8")
    rows = read_yolo_txt(label_txt)

    original_boxes = []
    work_boxes = []

    for row in rows:
        class_id = int(row[0])

        if class_id != PERSON_CLASS_ID:
            continue

        original_box = yolo_to_xyxy(row, img_w, img_h)
        original_box = clip_box_xyxy(original_box, img_w, img_h)

        work_box = expand_box_xyxy(original_box, img_w, img_h, BOX_SCALE)

        original_boxes.append(original_box)
        work_boxes.append(work_box)

    if not work_boxes:
        return 0

    results = sam_model.predict(
        source=image_rgb,
        bboxes=[box.tolist() for box in work_boxes],
        retina_masks=RETINA_MASKS,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]

    if result.masks is None:
        return 0

    masks = result.masks.data.cpu().numpy().astype(bool)

    count_saved = 0
    out_prefix = safe_blob_prefix(input_container_name, image_blob_name)

    for i, mask in enumerate(masks):
        original_box = original_boxes[i]
        work_box = work_boxes[i]

        # SAM получает work_box, но итоговую mask ограничиваем тем же work_box,
        # чтобы не сохранять островки вне crop.
        mask = keep_mask_inside_box(mask, work_box)

        crop_bgr = crop_by_box(image_bgr, work_box)
        crop_mask = mask_to_crop(mask, work_box, img_w, img_h)

        if crop_bgr.size == 0 or crop_mask.size == 0:
            continue

        item_prefix = f"{out_prefix}__person_{i:04d}"

        crop_blob_name = item_prefix + CROP_SUFFIX
        mask_blob_name = item_prefix + MASK_SUFFIX
        meta_blob_name = item_prefix + META_SUFFIX

        if not OVERWRITE and blob_exists(output_container, crop_blob_name):
            continue

        crop_png = encode_png(crop_bgr)
        mask_png = encode_png(crop_mask)

        upload_bytes(output_container, crop_blob_name, crop_png, "image/png")
        upload_bytes(output_container, mask_blob_name, mask_png, "image/png")

        if SAVE_META_JSON:
            meta = {
                "source_container": input_container_name,
                "source_image_blob": image_blob_name,
                "source_label_blob": label_blob_name,
                "person_index": i,
                "box_scale": BOX_SCALE,
                "image_size": {
                    "width": img_w,
                    "height": img_h,
                },
                "original_box_xyxy": [float(x) for x in original_box],
                "work_box_xyxy": [float(x) for x in work_box],
                "crop_blob": crop_blob_name,
                "mask_blob": mask_blob_name,
            }

            upload_bytes(
                output_container,
                meta_blob_name,
                json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
            )

        count_saved += 1

    return count_saved


def process_container(
    sam_model: SAM,
    service_client: BlobServiceClient,
    input_container_name: str,
    output_container_name: str,
):
    input_container = service_client.get_container_client(input_container_name)
    output_container = service_client.get_container_client(output_container_name)

    blob_names = [b.name for b in input_container.list_blobs()]
    blob_name_set = set(blob_names)

    image_names = [
        name for name in blob_names
        if is_image_blob(name) and label_name_for_image(name) in blob_name_set
    ]

    total_saved = 0

    for image_blob_name in tqdm(image_names, desc=f"Container {input_container_name}"):
        label_blob_name = label_name_for_image(image_blob_name)

        try:
            saved = process_one_image(
                sam_model=sam_model,
                input_container_name=input_container_name,
                input_container=input_container,
                output_container=output_container,
                image_blob_name=image_blob_name,
                label_blob_name=label_blob_name,
            )

            total_saved += saved

        except Exception as e:
            print(f"[ERROR] {input_container_name}/{image_blob_name}: {e}")

    print(f"[DONE] {input_container_name}: saved {total_saved} person crops/masks")


def main():
    service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    try:
        service_client.create_container(OUTPUT_CONTAINER)
        print(f"Created output container: {OUTPUT_CONTAINER}")
    except Exception:
        print(f"Output container exists or cannot be created: {OUTPUT_CONTAINER}")

    sam_model = SAM(SAM_MODEL_PATH)

    for container_name in INPUT_CONTAINERS:
        process_container(
            sam_model=sam_model,
            service_client=service_client,
            input_container_name=container_name,
            output_container_name=OUTPUT_CONTAINER,
        )


if __name__ == "__main__":
    main()