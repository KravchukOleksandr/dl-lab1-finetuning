import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_and_selector import (
    MicroNeXtMaskNet,
    SimplePointSelectorConfig,
    select_points_simple,
    point_inside_mask,
)


# =========================
# CONFIG
# =========================

DATASET_DIR = Path("data/student")
CHECKPOINT_PATH = Path("runs/student_mask/exp_simple_points_v1/best.pt")
OUTPUT_DIR = Path("runs/student_mask/exp_simple_points_v1/threshold_validation")

INPUT_W = 64
INPUT_H = 128

BATCH_SIZE = 256
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLDS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
]

SELECTOR_BASE_CFG = SimplePointSelectorConfig(
    mask_threshold=0.35,
    torso_y_min=0.15,
    torso_y_max=0.50,
    max_points=5,
)


# =========================
# DATASET
# =========================

class StudentMaskDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.rows = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        target = np.load(row["target_path"]).astype(np.float32)
        teacher_mask = np.load(row["teacher_mask_path"]).astype(np.uint8)

        image = image_rgb.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        target = target[None, :, :]
        teacher_mask = teacher_mask[None, :, :].astype(np.float32)

        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float(),
            "teacher_mask": torch.from_numpy(teacher_mask).float(),
            "sample_id": row["sample_id"],
        }


# =========================
# HELPERS
# =========================

def tensor_image_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def empty_point_metrics():
    return {
        "samples": 0,
        "total_selected_points": 0,
        "total_inside_points": 0,
        "valid_5_points": 0,
        "inside_count_sum": 0,
        "at_least": [0, 0, 0, 0, 0, 0],
    }


def update_point_metrics(acc: dict, points, teacher_mask: np.ndarray):
    inside_count = 0

    for p in points:
        if point_inside_mask(p, teacher_mask):
            inside_count += 1

    acc["samples"] += 1
    acc["total_selected_points"] += len(points)
    acc["total_inside_points"] += inside_count
    acc["inside_count_sum"] += inside_count

    if len(points) >= 5:
        acc["valid_5_points"] += 1

    for n in range(1, 6):
        if inside_count >= n:
            acc["at_least"][n] += 1


def finalize_point_metrics(acc: dict) -> dict:
    samples = max(1, acc["samples"])
    total_selected = max(1, acc["total_selected_points"])

    return {
        "valid_5_points_rate": acc["valid_5_points"] / samples,
        "selected_points_per_sample": acc["total_selected_points"] / samples,
        "inside_points_per_sample": acc["inside_count_sum"] / samples,
        "point_inside_rate": acc["total_inside_points"] / total_selected,
        "at_least_1_inside_rate": acc["at_least"][1] / samples,
        "at_least_2_inside_rate": acc["at_least"][2] / samples,
        "at_least_3_inside_rate": acc["at_least"][3] / samples,
        "at_least_4_inside_rate": acc["at_least"][4] / samples,
        "at_least_5_inside_rate": acc["at_least"][5] / samples,
    }


# =========================
# VALIDATION
# =========================

@torch.no_grad()
def evaluate_threshold(model, loader, threshold: float):
    model.eval()

    cfg = replace(SELECTOR_BASE_CFG, mask_threshold=threshold)

    acc = empty_point_metrics()

    for batch in tqdm(loader, desc=f"threshold={threshold:.2f}", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        teacher_masks = batch["teacher_mask"].to(DEVICE, non_blocking=True)

        logits = model(images)
        pred = torch.sigmoid(logits)

        pred_up = F.interpolate(
            pred,
            size=(INPUT_H, INPUT_W),
            mode="bilinear",
            align_corners=False,
        )

        bs = images.shape[0]

        for i in range(bs):
            image_np = tensor_image_to_uint8_rgb(images[i])
            pred_np = pred_up[i, 0].detach().cpu().numpy()
            teacher_np = teacher_masks[i, 0].detach().cpu().numpy()

            points = select_points_simple(
                image_rgb=image_np,
                pred_mask=pred_np,
                cfg=cfg,
            )

            update_point_metrics(acc, points, teacher_np)

    row = {"threshold": threshold}
    row.update(finalize_point_metrics(acc))
    return row


def save_plot(rows):
    thresholds = [r["threshold"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["at_least_1_inside_rate"] for r in rows], marker="o", label=">=1 inside")
    plt.plot(thresholds, [r["at_least_2_inside_rate"] for r in rows], marker="o", label=">=2 inside")
    plt.plot(thresholds, [r["at_least_3_inside_rate"] for r in rows], marker="o", label=">=3 inside")
    plt.plot(thresholds, [r["at_least_4_inside_rate"] for r in rows], marker="o", label=">=4 inside")
    plt.plot(thresholds, [r["at_least_5_inside_rate"] for r in rows], marker="o", label=">=5 inside")

    plt.xlabel("mask_threshold")
    plt.ylabel("rate")
    plt.title("Point-inside-mask metrics by threshold")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_metrics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["selected_points_per_sample"] for r in rows], marker="o", label="selected points/sample")
    plt.plot(thresholds, [r["inside_points_per_sample"] for r in rows], marker="o", label="inside points/sample")
    plt.xlabel("mask_threshold")
    plt.ylabel("points")
    plt.title("Selected and inside points by threshold")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_points_count.png", dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_dataset = StudentMaskDataset(DATASET_DIR / "manifest_val.csv")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    model = MicroNeXtMaskNet().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch:  {checkpoint.get('epoch', 'unknown')}")
    print(f"Val samples:       {len(val_dataset)}")
    print(f"Device:            {DEVICE}")

    rows = []

    for threshold in THRESHOLDS:
        row = evaluate_threshold(model, val_loader, threshold)
        rows.append(row)

        print()
        print(f"threshold={threshold:.2f}")
        print(f"  valid_5_points_rate:        {row['valid_5_points_rate']:.4f}")
        print(f"  selected_points_per_sample: {row['selected_points_per_sample']:.2f}")
        print(f"  inside_points_per_sample:   {row['inside_points_per_sample']:.2f}")
        print(f"  point_inside_rate:          {row['point_inside_rate']:.4f}")
        print(f"  at_least_1_inside_rate:     {row['at_least_1_inside_rate']:.4f}")
        print(f"  at_least_2_inside_rate:     {row['at_least_2_inside_rate']:.4f}")
        print(f"  at_least_3_inside_rate:     {row['at_least_3_inside_rate']:.4f}")
        print(f"  at_least_4_inside_rate:     {row['at_least_4_inside_rate']:.4f}")
        print(f"  at_least_5_inside_rate:     {row['at_least_5_inside_rate']:.4f}")

    csv_path = OUTPUT_DIR / "threshold_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_plot(rows)

    best = max(rows, key=lambda r: r["at_least_4_inside_rate"])

    summary = {
        "best_by_at_least_4_inside_rate": best,
        "checkpoint": str(CHECKPOINT_PATH),
        "thresholds": THRESHOLDS,
    }

    with open(OUTPUT_DIR / "threshold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Best threshold by at_least_4_inside_rate:")
    print(json.dumps(best, indent=2))
    print()
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved plots: {OUTPUT_DIR / 'threshold_metrics.png'}")
    print(f"             {OUTPUT_DIR / 'threshold_points_count.png'}")


if __name__ == "__main__":
    main()