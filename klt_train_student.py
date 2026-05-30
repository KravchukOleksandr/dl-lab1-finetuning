import csv
import json
import math
import random
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
    PointSelectorConfig,
    select_triangle_points,
    triangle_area,
    min_pairwise_distance,
)


# =========================
# CONFIG
# =========================

DATASET_DIR = Path("data/student")
RUN_DIR = Path("runs/student_mask/exp001")

INPUT_W = 64
INPUT_H = 128
TARGET_W = 16
TARGET_H = 32

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

SAVE_VIS_EVERY = 5
VIS_SAMPLES = 8

BEST_METRIC_NAME = "all_3_points_inside_teacher_mask_rate"

# Point selector config
SELECTOR_CFG = PointSelectorConfig(
    mask_threshold=0.35,
    torso_y_min=0.25,
    torso_y_max=0.70,
    fat_row_ratio=0.75,
    tri_top_y_frac=0.18,
    tri_low_y_frac=0.16,
    tri_side_x_frac=0.22,
    micro_square_frac=0.28,
    min_micro_half=3,
    score_mask_power=1.3,
    min_triangle_area=18.0,
    min_point_distance=4.0,
)

AUG_HFLIP = True
AUG_BRIGHTNESS_CONTRAST = True


# =========================
# DATASET
# =========================

class StudentMaskDataset(Dataset):
    def __init__(self, manifest_path, train=False):
        self.rows = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        target = np.load(row["target_path"]).astype(np.float32)
        teacher_mask = np.load(row["teacher_mask_path"]).astype(np.uint8)

        if self.train and AUG_HFLIP and random.random() < 0.5:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            target = np.ascontiguousarray(target[:, ::-1])
            teacher_mask = np.ascontiguousarray(teacher_mask[:, ::-1])

        if self.train and AUG_BRIGHTNESS_CONTRAST:
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-12.0, 12.0)
                image_rgb = np.clip(image_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_image_to_uint8_rgb(t):
    img = t.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_checkpoint(path, model, optimizer, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def append_metrics_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def point_inside_mask(point, mask):
    x, y = point
    x = int(round(x))
    y = int(round(y))

    h, w = mask.shape

    if x < 0 or y < 0 or x >= w or y >= h:
        return False

    return bool(mask[y, x] > 0.5)


# =========================
# VALIDATION
# =========================

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_samples = 0
    valid_triangles = 0

    total_selected_points = 0
    points_inside_teacher = 0
    all_3_inside_teacher = 0

    triangle_areas = []
    min_distances = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        targets = batch["target"].to(DEVICE, non_blocking=True)
        teacher_masks = batch["teacher_mask"].to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += float(loss.item())
        total_batches += 1

        pred = torch.sigmoid(logits)
        pred_up = F.interpolate(
            pred,
            size=(INPUT_H, INPUT_W),
            mode="bilinear",
            align_corners=False,
        )

        batch_size = images.shape[0]

        for i in range(batch_size):
            image_np = tensor_image_to_uint8_rgb(images[i])
            pred_mask_np = pred_up[i, 0].detach().cpu().numpy()
            teacher_mask_np = teacher_masks[i, 0].detach().cpu().numpy()

            points = select_triangle_points(
                image_rgb=image_np,
                pred_mask=pred_mask_np,
                cfg=SELECTOR_CFG,
            )

            total_samples += 1

            if len(points) == 3:
                valid_triangles += 1
                triangle_areas.append(triangle_area(points))
                min_distances.append(min_pairwise_distance(points))

            all_inside = len(points) == 3

            for p in points:
                inside = point_inside_mask(p, teacher_mask_np)

                total_selected_points += 1
                points_inside_teacher += int(inside)

                if not inside:
                    all_inside = False

            if all_inside:
                all_3_inside_teacher += 1

    return {
        "val_loss": total_loss / max(1, total_batches),
        "valid_triangle_rate": valid_triangles / max(1, total_samples),
        "point_inside_teacher_mask_rate": points_inside_teacher / max(1, total_selected_points),
        "all_3_points_inside_teacher_mask_rate": all_3_inside_teacher / max(1, total_samples),
        "mean_triangle_area": float(np.mean(triangle_areas)) if triangle_areas else 0.0,
        "mean_min_point_distance": float(np.mean(min_distances)) if min_distances else 0.0,
        "selected_points_per_sample": total_selected_points / max(1, total_samples),
    }


@torch.no_grad()
def save_visualizations(model, loader, epoch):
    model.eval()

    vis_dir = RUN_DIR / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))

    images = batch["image"].to(DEVICE)
    teacher_masks = batch["teacher_mask"].to(DEVICE)

    logits = model(images)
    pred = torch.sigmoid(logits)

    pred_up = F.interpolate(
        pred,
        size=(INPUT_H, INPUT_W),
        mode="bilinear",
        align_corners=False,
    )

    n = min(VIS_SAMPLES, images.shape[0])

    plt.figure(figsize=(12, 4 * n))

    for i in range(n):
        image_np = tensor_image_to_uint8_rgb(images[i])
        teacher_np = teacher_masks[i, 0].cpu().numpy()
        pred_np = pred_up[i, 0].cpu().numpy()

        points = select_triangle_points(
            image_rgb=image_np,
            pred_mask=pred_np,
            cfg=SELECTOR_CFG,
        )

        overlay_teacher = image_np.copy()
        overlay_pred = image_np.copy()

        red = np.array([255, 0, 0], dtype=np.float32)
        blue = np.array([0, 80, 255], dtype=np.float32)

        teacher_bool = teacher_np > 0.5
        pred_bool = pred_np > SELECTOR_CFG.mask_threshold

        overlay_teacher = overlay_teacher.astype(np.float32)
        overlay_pred = overlay_pred.astype(np.float32)

        overlay_teacher[teacher_bool] = 0.55 * overlay_teacher[teacher_bool] + 0.45 * red
        overlay_pred[pred_bool] = 0.55 * overlay_pred[pred_bool] + 0.45 * blue

        overlay_teacher = overlay_teacher.astype(np.uint8)
        overlay_pred = overlay_pred.astype(np.uint8)

        triangle_img = image_np.copy()

        if len(points) == 3:
            pts = np.array(points, dtype=np.int32)

            for x, y in pts:
                cv2.circle(triangle_img, (int(x), int(y)), 3, (0, 255, 0), -1)

            cv2.line(triangle_img, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 1)
            cv2.line(triangle_img, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 1)
            cv2.line(triangle_img, tuple(pts[2]), tuple(pts[0]), (0, 255, 0), 1)

        ax = plt.subplot(n, 4, i * 4 + 1)
        ax.imshow(image_np)
        ax.set_title("Crop")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 2)
        ax.imshow(overlay_teacher)
        ax.set_title("Teacher mask")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 3)
        ax.imshow(overlay_pred)
        ax.set_title("Pred mask")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 4)
        ax.imshow(triangle_img)
        ax.set_title("Triangle points")
        ax.axis("off")

    plt.tight_layout()
    out_path = vis_dir / f"epoch_{epoch:04d}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# TRAIN
# =========================

def main():
    set_seed(RANDOM_SEED)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "INPUT_W": INPUT_W,
        "INPUT_H": INPUT_H,
        "TARGET_W": TARGET_W,
        "TARGET_H": TARGET_H,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DEVICE": DEVICE,
        "USE_AMP": USE_AMP,
        "selector_cfg": SELECTOR_CFG.__dict__,
    }

    with open(RUN_DIR / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    train_dataset = StudentMaskDataset(
        DATASET_DIR / "manifest_train.csv",
        train=True,
    )

    val_dataset = StudentMaskDataset(
        DATASET_DIR / "manifest_val.csv",
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    model = MicroNeXtMaskNet().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Total params:  {total_params:,}")
    print(f"Trainable:     {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(USE_AMP and DEVICE == "cuda")
    )

    best_metric = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            images = batch["image"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        train_loss = train_loss_sum / max(1, train_batches)

        val_metrics = validate(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            **val_metrics,
        }

        append_metrics_csv(RUN_DIR / "metrics.csv", row)

        metric_value = val_metrics[BEST_METRIC_NAME]

        is_best = metric_value > best_metric

        if is_best:
            best_metric = metric_value
            save_checkpoint(
                RUN_DIR / "best.pt",
                model,
                optimizer,
                epoch,
                row,
            )

        save_checkpoint(
            RUN_DIR / "last.pt",
            model,
            optimizer,
            epoch,
            row,
        )

        if epoch == 1 or epoch % SAVE_VIS_EVERY == 0:
            save_visualizations(model, val_loader, epoch)

        print()
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  train_loss: {train_loss:.5f}")
        print(f"  val_loss:   {val_metrics['val_loss']:.5f}")
        print(f"  valid_triangle_rate:                  {val_metrics['valid_triangle_rate']:.4f}")
        print(f"  point_inside_teacher_mask_rate:        {val_metrics['point_inside_teacher_mask_rate']:.4f}")
        print(f"  all_3_points_inside_teacher_mask_rate: {val_metrics['all_3_points_inside_teacher_mask_rate']:.4f}")
        print(f"  mean_triangle_area:                    {val_metrics['mean_triangle_area']:.2f}")
        print(f"  mean_min_point_distance:               {val_metrics['mean_min_point_distance']:.2f}")
        print(f"  best {BEST_METRIC_NAME}:               {best_metric:.4f}")
        print(f"  saved best:                            {is_best}")
        print()

    print(f"Training finished. Best checkpoint: {RUN_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()