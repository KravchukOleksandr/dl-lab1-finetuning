import csv
import json
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
    SimplePointSelectorConfig,
    select_points_simple,
    point_inside_mask,
)


# =========================
# CONFIG
# =========================

DATASET_DIR = Path("data/student")
RUN_DIR = Path("runs/student_mask/exp_simple_points_v1")

INPUT_W = 64
INPUT_H = 128
TARGET_W = 16
TARGET_H = 32

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

SAVE_VIS_EVERY = 2
VIS_SAMPLES = 12

BEST_METRIC_NAME = "at_least_4_inside_rate"

SELECTOR_CFG = SimplePointSelectorConfig(
    mask_threshold=0.35,
    torso_y_min=0.15,
    torso_y_max=0.50,
    max_points=5,
)

RUN_SELECTOR_CEILING_AT_START = True

AUG_HFLIP = True
AUG_BRIGHTNESS_CONTRAST = True


# =========================
# DATASET
# =========================

class StudentMaskDataset(Dataset):
    def __init__(self, manifest_path: Path, train: bool = False):
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

        if self.train and AUG_BRIGHTNESS_CONTRAST and random.random() < 0.5:
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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_image_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict):
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


def append_metrics_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def empty_point_metrics():
    return {
        "samples": 0,
        "total_selected_points": 0,
        "total_inside_points": 0,
        "valid_5_points": 0,
        "inside_count_sum": 0,
        "at_least": [0, 0, 0, 0, 0, 0],  # индекс 1..5
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


def finalize_point_metrics(acc: dict, prefix: str = "") -> dict:
    samples = max(1, acc["samples"])
    total_selected = max(1, acc["total_selected_points"])

    p = f"{prefix}_" if prefix else ""

    return {
        f"{p}valid_5_points_rate": acc["valid_5_points"] / samples,
        f"{p}selected_points_per_sample": acc["total_selected_points"] / samples,
        f"{p}inside_points_per_sample": acc["inside_count_sum"] / samples,
        f"{p}point_inside_rate": acc["total_inside_points"] / total_selected,
        f"{p}at_least_1_inside_rate": acc["at_least"][1] / samples,
        f"{p}at_least_2_inside_rate": acc["at_least"][2] / samples,
        f"{p}at_least_3_inside_rate": acc["at_least"][3] / samples,
        f"{p}at_least_4_inside_rate": acc["at_least"][4] / samples,
        f"{p}at_least_5_inside_rate": acc["at_least"][5] / samples,
    }


# =========================
# CEILING CHECK
# =========================

@torch.no_grad()
def evaluate_selector_ceiling(loader):
    """
    Проверка потолка selector-а:
    1. selector на teacher mask 64x128
    2. selector на target 16x32, поднятом обратно до 64x128

    Если target_up сильно хуже teacher, значит 16x32 может быть узким местом.
    """
    teacher_acc = empty_point_metrics()
    target_up_acc = empty_point_metrics()

    for batch in tqdm(loader, desc="Selector ceiling"):
        images = batch["image"].to(DEVICE, non_blocking=True)
        targets = batch["target"].to(DEVICE, non_blocking=True)
        teacher_masks = batch["teacher_mask"].to(DEVICE, non_blocking=True)

        target_up = F.interpolate(
            targets,
            size=(INPUT_H, INPUT_W),
            mode="bilinear",
            align_corners=False,
        )

        bs = images.shape[0]

        for i in range(bs):
            image_np = tensor_image_to_uint8_rgb(images[i])
            teacher_np = teacher_masks[i, 0].cpu().numpy()
            target_np = target_up[i, 0].cpu().numpy()

            teacher_points = select_points_simple(
                image_rgb=image_np,
                pred_mask=teacher_np,
                cfg=SELECTOR_CFG,
            )

            target_points = select_points_simple(
                image_rgb=image_np,
                pred_mask=target_np,
                cfg=SELECTOR_CFG,
            )

            update_point_metrics(teacher_acc, teacher_points, teacher_np)
            update_point_metrics(target_up_acc, target_points, teacher_np)

    metrics = {}
    metrics.update(finalize_point_metrics(teacher_acc, "teacher"))
    metrics.update(finalize_point_metrics(target_up_acc, "target_up"))

    return metrics


# =========================
# VALIDATION
# =========================

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    point_acc = empty_point_metrics()

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

        bs = images.shape[0]

        for i in range(bs):
            image_np = tensor_image_to_uint8_rgb(images[i])
            pred_np = pred_up[i, 0].detach().cpu().numpy()
            teacher_np = teacher_masks[i, 0].detach().cpu().numpy()

            points = select_points_simple(
                image_rgb=image_np,
                pred_mask=pred_np,
                cfg=SELECTOR_CFG,
            )

            update_point_metrics(point_acc, points, teacher_np)

    metrics = {
        "val_loss": total_loss / max(1, total_batches),
    }

    metrics.update(finalize_point_metrics(point_acc))

    return metrics


# =========================
# VISUALIZATION
# =========================

@torch.no_grad()
def save_visualizations(model, loader, epoch: int):
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

        points = select_points_simple(
            image_rgb=image_np,
            pred_mask=pred_np,
            cfg=SELECTOR_CFG,
        )

        teacher_overlay = image_np.copy().astype(np.float32)
        pred_overlay = image_np.copy().astype(np.float32)
        points_img = image_np.copy()

        red = np.array([255, 0, 0], dtype=np.float32)
        blue = np.array([0, 80, 255], dtype=np.float32)

        teacher_bool = teacher_np > 0.5
        pred_bool = pred_np > SELECTOR_CFG.mask_threshold

        teacher_overlay[teacher_bool] = 0.55 * teacher_overlay[teacher_bool] + 0.45 * red
        pred_overlay[pred_bool] = 0.55 * pred_overlay[pred_bool] + 0.45 * blue

        teacher_overlay = teacher_overlay.astype(np.uint8)
        pred_overlay = pred_overlay.astype(np.uint8)

        for x, y in points:
            color = (0, 255, 0) if point_inside_mask((x, y), teacher_np) else (255, 0, 0)
            cv2.circle(points_img, (int(round(x)), int(round(y))), 3, color, -1)

        ax = plt.subplot(n, 4, i * 4 + 1)
        ax.imshow(image_np)
        ax.set_title("Crop")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 2)
        ax.imshow(teacher_overlay)
        ax.set_title("Teacher mask")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 3)
        ax.imshow(pred_overlay)
        ax.set_title("Pred mask")
        ax.axis("off")

        ax = plt.subplot(n, 4, i * 4 + 4)
        ax.imshow(points_img)
        ax.set_title("Selected 5 points")
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
        "BEST_METRIC_NAME": BEST_METRIC_NAME,
        "selector_cfg": SELECTOR_CFG.__dict__,
    }

    with open(RUN_DIR / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    train_dataset = StudentMaskDataset(DATASET_DIR / "manifest_train.csv", train=True)
    val_dataset = StudentMaskDataset(DATASET_DIR / "manifest_val.csv", train=False)

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

    print(f"Device:        {DEVICE}")
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

    use_amp = USE_AMP and DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if RUN_SELECTOR_CEILING_AT_START:
        print()
        print("Checking selector ceiling...")
        ceiling_metrics = evaluate_selector_ceiling(val_loader)

        with open(RUN_DIR / "selector_ceiling.json", "w", encoding="utf-8") as f:
            json.dump(ceiling_metrics, f, indent=2)

        print("Selector ceiling:")
        print(f"  teacher_at_least_4_inside_rate:  {ceiling_metrics['teacher_at_least_4_inside_rate']:.4f}")
        print(f"  teacher_at_least_5_inside_rate:  {ceiling_metrics['teacher_at_least_5_inside_rate']:.4f}")
        print(f"  target_up_at_least_4_inside_rate:{ceiling_metrics['target_up_at_least_4_inside_rate']:.4f}")
        print(f"  target_up_at_least_5_inside_rate:{ceiling_metrics['target_up_at_least_5_inside_rate']:.4f}")
        print()

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

            with torch.cuda.amp.autocast(enabled=use_amp):
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
            save_checkpoint(RUN_DIR / "best.pt", model, optimizer, epoch, row)

        save_checkpoint(RUN_DIR / "last.pt", model, optimizer, epoch, row)

        if epoch == 1 or epoch % SAVE_VIS_EVERY == 0:
            save_visualizations(model, val_loader, epoch)

        print()
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  train_loss: {train_loss:.5f}")
        print(f"  val_loss:   {val_metrics['val_loss']:.5f}")
        print(f"  valid_5_points_rate:          {val_metrics['valid_5_points_rate']:.4f}")
        print(f"  selected_points_per_sample:   {val_metrics['selected_points_per_sample']:.2f}")
        print(f"  inside_points_per_sample:     {val_metrics['inside_points_per_sample']:.2f}")
        print(f"  point_inside_rate:            {val_metrics['point_inside_rate']:.4f}")
        print(f"  at_least_1_inside_rate:       {val_metrics['at_least_1_inside_rate']:.4f}")
        print(f"  at_least_2_inside_rate:       {val_metrics['at_least_2_inside_rate']:.4f}")
        print(f"  at_least_3_inside_rate:       {val_metrics['at_least_3_inside_rate']:.4f}")
        print(f"  at_least_4_inside_rate:       {val_metrics['at_least_4_inside_rate']:.4f}")
        print(f"  at_least_5_inside_rate:       {val_metrics['at_least_5_inside_rate']:.4f}")
        print(f"  best {BEST_METRIC_NAME}:      {best_metric:.4f}")
        print(f"  saved best:                   {is_best}")
        print()

    print("Training finished.")
    print(f"Best checkpoint: {RUN_DIR / 'best.pt'}")
    print(f"Last checkpoint: {RUN_DIR / 'last.pt'}")
    print(f"Metrics:         {RUN_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()