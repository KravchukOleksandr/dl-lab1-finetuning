import csv
import io
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient


# =========================
# CONFIG
# =========================

AZURE_CONNECTION_STRING = "PUT_CONNECTION_STRING_HERE"
CONTAINER_NAME = "sam-crops-masks"

LOCAL_DATASET_DIR = Path("data/student")

INPUT_W = 64
INPUT_H = 128

TARGET_W = 16
TARGET_H = 32

ASPECT_MIN = 0.10
ASPECT_MAX = 1.00
MIN_DIAGONAL = 45.0

VAL_FRACTION = 0.10
RANDOM_SEED = 42

IMAGE_PAD_VALUE = 114
MASK_PAD_VALUE = 0

CLEAR_OUTPUT_DIR = True

META_SUFFIX = "_meta.json"


# =========================
# UTILS
# =========================

def box_width_height(box):
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width, height


def infer_pair_blobs(meta_blob_name):
    base = meta_blob_name[:-len(META_SUFFIX)]
    return base + "_crop.png", base + "_mask.png"


def download_blob_bytes(container, blob_name):
    return container.get_blob_client(blob_name).download_blob().readall()


def decode_image_bgr(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def decode_mask_gray(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Cannot decode mask")
    return mask


def letterbox_image(image, out_w, out_h, pad_value):
    h, w = image.shape[:2]

    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)

    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    transform = {
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }

    return canvas, transform


def letterbox_mask_with_transform(mask, out_w, out_h, transform, pad_value=0):
    new_w = transform["new_w"]
    new_h = transform["new_h"]
    pad_x = transform["pad_x"]
    pad_y = transform["pad_y"]

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((out_h, out_w), pad_value, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def downsample_mask_area(mask_128x64):
    mask_float = (mask_128x64.astype(np.float32) / 255.0)
    target = cv2.resize(mask_float, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    target = np.clip(target, 0.0, 1.0).astype(np.float32)
    return target


def gaussian_kde_manual(values, points=400):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return None, None

    v_min = values.min()
    v_max = values.max()

    if v_min == v_max:
        return None, None

    xs = np.linspace(v_min, v_max, points)

    std = values.std(ddof=1)
    n = len(values)
    bandwidth = 1.06 * std * (n ** (-1 / 5))

    if bandwidth <= 1e-9:
        return None, None

    diffs = (xs[:, None] - values[None, :]) / bandwidth
    ys = np.exp(-0.5 * diffs ** 2).sum(axis=1)
    ys /= n * bandwidth * math.sqrt(2 * math.pi)

    return xs, ys


def plot_density(values, title, xlabel, output_path):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=50,
        density=True,
        alpha=0.35,
        edgecolor="black",
        label="Histogram density",
    )

    xs, ys = gaussian_kde_manual(values)
    if xs is not None:
        plt.plot(xs, ys, linewidth=2, label="KDE")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_manifest(path, rows):
    fieldnames = [
        "sample_id",
        "split",
        "image_path",
        "target_path",
        "teacher_mask_path",
        "source_container",
        "source_image_blob",
        "source_label_blob",
        "crop_blob",
        "mask_blob",
        "meta_blob",
        "aspect_ratio",
        "diagonal",
        "work_box_xyxy",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =========================
# MAIN
# =========================

def main():
    random.seed(RANDOM_SEED)

    if CLEAR_OUTPUT_DIR and LOCAL_DATASET_DIR.exists():
        shutil.rmtree(LOCAL_DATASET_DIR)

    train_img_dir = LOCAL_DATASET_DIR / "train/images"
    train_target_dir = LOCAL_DATASET_DIR / "train/targets"
    train_teacher_dir = LOCAL_DATASET_DIR / "train/teacher_masks"

    val_img_dir = LOCAL_DATASET_DIR / "val/images"
    val_target_dir = LOCAL_DATASET_DIR / "val/targets"
    val_teacher_dir = LOCAL_DATASET_DIR / "val/teacher_masks"

    for d in [
        train_img_dir, train_target_dir, train_teacher_dir,
        val_img_dir, val_target_dir, val_teacher_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)

    meta_blobs = [
        blob.name
        for blob in container.list_blobs()
        if blob.name.endswith(META_SUFFIX)
    ]

    print(f"Found meta files: {len(meta_blobs)}")

    candidates = []
    aspect_values = []
    diagonal_values = []

    for meta_blob in tqdm(meta_blobs, desc="Scanning meta"):
        data = download_blob_bytes(container, meta_blob)
        meta = json.loads(data.decode("utf-8"))

        if "work_box_xyxy" not in meta:
            continue

        width, height = box_width_height(meta["work_box_xyxy"])

        if width <= 0 or height <= 0:
            continue

        aspect = width / height
        diagonal = math.sqrt(width * width + height * height)

        aspect_values.append(aspect)
        diagonal_values.append(diagonal)

        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        if diagonal < MIN_DIAGONAL:
            continue

        crop_blob = meta.get("crop_blob")
        mask_blob = meta.get("mask_blob")

        if crop_blob is None or mask_blob is None:
            crop_blob, mask_blob = infer_pair_blobs(meta_blob)

        candidates.append({
            "meta": meta,
            "meta_blob": meta_blob,
            "crop_blob": crop_blob,
            "mask_blob": mask_blob,
            "aspect_ratio": aspect,
            "diagonal": diagonal,
        })

    print(f"Candidates after filters: {len(candidates)}")

    plot_density(
        aspect_values,
        "Aspect Ratio Density, work_box width / height",
        "width / height",
        LOCAL_DATASET_DIR / "aspect_ratio_density.png",
    )

    plot_density(
        diagonal_values,
        "Work Box Diagonal Size Density",
        "diagonal size, px",
        LOCAL_DATASET_DIR / "diagonal_density.png",
    )

    random.shuffle(candidates)

    val_count = int(round(len(candidates) * VAL_FRACTION))
    val_ids = set(range(val_count))

    train_manifest = []
    val_manifest = []

    saved_train = 0
    saved_val = 0

    for idx, item in enumerate(tqdm(candidates, desc="Preparing dataset")):
        split = "val" if idx in val_ids else "train"

        sample_id = f"sample_{idx:07d}"

        if split == "train":
            img_dir = train_img_dir
            target_dir = train_target_dir
            teacher_dir = train_teacher_dir
        else:
            img_dir = val_img_dir
            target_dir = val_target_dir
            teacher_dir = val_teacher_dir

        image_out = img_dir / f"{sample_id}.png"
        target_out = target_dir / f"{sample_id}.npy"
        teacher_out = teacher_dir / f"{sample_id}.npy"

        try:
            crop_data = download_blob_bytes(container, item["crop_blob"])
            mask_data = download_blob_bytes(container, item["mask_blob"])

            crop_bgr = decode_image_bgr(crop_data)
            mask_gray = decode_mask_gray(mask_data)

            image_lb, transform = letterbox_image(
                crop_bgr,
                INPUT_W,
                INPUT_H,
                IMAGE_PAD_VALUE,
            )

            mask_lb = letterbox_mask_with_transform(
                mask_gray,
                INPUT_W,
                INPUT_H,
                transform,
                MASK_PAD_VALUE,
            )

            mask_lb = ((mask_lb > 127).astype(np.uint8) * 255)

            target = downsample_mask_area(mask_lb)
            teacher_mask = (mask_lb > 127).astype(np.uint8)

            cv2.imwrite(str(image_out), image_lb)
            np.save(str(target_out), target)
            np.save(str(teacher_out), teacher_mask)

        except Exception as e:
            print(f"[SKIP] {item['meta_blob']}: {e}")
            continue

        manifest_row = {
            "sample_id": sample_id,
            "split": split,
            "image_path": str(image_out),
            "target_path": str(target_out),
            "teacher_mask_path": str(teacher_out),
            "source_container": item["meta"].get("source_container", ""),
            "source_image_blob": item["meta"].get("source_image_blob", ""),
            "source_label_blob": item["meta"].get("source_label_blob", ""),
            "crop_blob": item["crop_blob"],
            "mask_blob": item["mask_blob"],
            "meta_blob": item["meta_blob"],
            "aspect_ratio": item["aspect_ratio"],
            "diagonal": item["diagonal"],
            "work_box_xyxy": json.dumps(item["meta"]["work_box_xyxy"]),
        }

        if split == "train":
            train_manifest.append(manifest_row)
            saved_train += 1
        else:
            val_manifest.append(manifest_row)
            saved_val += 1

    save_manifest(LOCAL_DATASET_DIR / "manifest_train.csv", train_manifest)
    save_manifest(LOCAL_DATASET_DIR / "manifest_val.csv", val_manifest)

    print()
    print(f"Saved train samples: {saved_train}")
    print(f"Saved val samples:   {saved_val}")
    print(f"Dataset path:        {LOCAL_DATASET_DIR.resolve()}")
    print(f"Aspect plot:         {(LOCAL_DATASET_DIR / 'aspect_ratio_density.png').resolve()}")
    print(f"Diagonal plot:       {(LOCAL_DATASET_DIR / 'diagonal_density.png').resolve()}")


if __name__ == "__main__":
    main()