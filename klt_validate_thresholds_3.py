import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

SELECTOR_BASE_CFG = SimplePointSelectorConfig(
    mask_threshold=0.80,
    torso_y_min=0.20,
    torso_y_max=0.55,
    max_points=5,
)

SAVE_FAIL_CASES = True
FAIL_CASE_THRESHOLD = 0.80
FAIL_IF_INSIDE_LESS_THAN = 4
MAX_FAIL_CASES = 300
FAIL_CASES_DIR = OUTPUT_DIR / "fail_cases"


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


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


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

    return inside_count


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


def save_fail_case_image(
    out_path: Path,
    image_rgb: np.ndarray,
    teacher_mask: np.ndarray,
    pred_mask: np.ndarray,
    points,
    threshold: float,
    inside_count: int,
    sample_id: str,
):
    teacher_overlay = image_rgb.copy().astype(np.float32)
    pred_overlay = image_rgb.copy().astype(np.float32)
    points_img = image_rgb.copy()

    red = np.array([255, 0, 0], dtype=np.float32)
    blue = np.array([0, 80, 255], dtype=np.float32)

    teacher_bool = teacher_mask > 0.5
    pred_bool = pred_mask > threshold

    teacher_overlay[teacher_bool] = 0.55 * teacher_overlay[teacher_bool] + 0.45 * red
    pred_overlay[pred_bool] = 0.55 * pred_overlay[pred_bool] + 0.45 * blue

    teacher_overlay = teacher_overlay.astype(np.uint8)
    pred_overlay = pred_overlay.astype(np.uint8)

    for p in points:
        x, y = p
        inside = point_inside_mask(p, teacher_mask)
        color = (0, 255, 0) if inside else (255, 0, 0)
        cv2.circle(points_img, (int(round(x)), int(round(y))), 3, color, -1)

    fig = plt.figure(figsize=(16, 4))

    ax = plt.subplot(1, 4, 1)
    ax.imshow(image_rgb)
    ax.set_title(f"Crop\n{sample_id}")
    ax.axis("off")

    ax = plt.subplot(1, 4, 2)
    ax.imshow(teacher_overlay)
    ax.set_title("Teacher mask")
    ax.axis("off")

    ax = plt.subplot(1, 4, 3)
    ax.imshow(pred_overlay)
    ax.set_title(f"Pred mask\nthr={threshold:.2f}")
    ax.axis("off")

    ax = plt.subplot(1, 4, 4)
    ax.imshow(points_img)
    ax.set_title(f"Points\ninside={inside_count}/5")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# =========================
# VALIDATION
# =========================

@torch.no_grad()
def evaluate_threshold(model, loader, threshold: float, save_fails: bool = False):
    model.eval()

    cfg = replace(SELECTOR_BASE_CFG, mask_threshold=threshold)
    acc = empty_point_metrics()

    fail_rows = []
    fail_saved = 0

    if save_fails:
        FAIL_CASES_DIR.mkdir(parents=True, exist_ok=True)

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
            sample_id = batch["sample_id"][i]

            points = select_points_simple(
                image_rgb=image_np,
                pred_mask=pred_np,
                cfg=cfg,
            )

            inside_count = update_point_metrics(acc, points, teacher_np)

            if (
                save_fails
                and inside_count < FAIL_IF_INSIDE_LESS_THAN
                and fail_saved < MAX_FAIL_CASES
            ):
                filename = (
                    f"fail_{fail_saved:04d}_"
                    f"{safe_filename(sample_id)}_"
                    f"inside_{inside_count}_"
                    f"thr_{threshold:.2f}.png"
                )

                out_path = FAIL_CASES_DIR / filename

                save_fail_case_image(
                    out_path=out_path,
                    image_rgb=image_np,
                    teacher_mask=teacher_np,
                    pred_mask=pred_np,
                    points=points,
                    threshold=threshold,
                    inside_count=inside_count,
                    sample_id=sample_id,
                )

                fail_rows.append({
                    "sample_id": sample_id,
                    "threshold": threshold,
                    "inside_count": inside_count,
                    "points_count": len(points),
                    "image": str(out_path),
                })

                fail_saved += 1

    row = {"threshold": threshold}
    row.update(finalize_point_metrics(acc))

    return row, fail_rows


def save_plots(rows):
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
    all_fail_rows = []

    for threshold in THRESHOLDS:
        save_fails = SAVE_FAIL_CASES and abs(threshold - FAIL_CASE_THRESHOLD) < 1e-9

        row, fail_rows = evaluate_threshold(
            model=model,
            loader=val_loader,
            threshold=threshold,
            save_fails=save_fails,
        )

        rows.append(row)
        all_fail_rows.extend(fail_rows)

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

    if all_fail_rows:
        fail_csv = FAIL_CASES_DIR / "fail_cases.csv"

        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_fail_rows[0].keys()))
            writer.writeheader()
            for row in all_fail_rows:
                writer.writerow(row)

    save_plots(rows)

    best = max(rows, key=lambda r: r["at_least_4_inside_rate"])

    summary = {
        "best_by_at_least_4_inside_rate": best,
        "checkpoint": str(CHECKPOINT_PATH),
        "thresholds": THRESHOLDS,
        "fail_cases_dir": str(FAIL_CASES_DIR) if SAVE_FAIL_CASES else None,
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

    if all_fail_rows:
        print(f"Saved fail cases: {FAIL_CASES_DIR}")


if __name__ == "__main__":
    main()