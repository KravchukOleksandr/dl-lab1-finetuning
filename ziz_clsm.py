"""
Quick Helmet vs No-Helmet classifier experiment (single script)

What it does:
1) Downloads images+labels from Azure Blob Storage containers (train+val given by you)
2) Extracts GT head crops:
   - class 1 -> helmet
   - class 2 -> no_helmet
3) Saves crops into:
   OUT_DIR/
     train/helmet, train/no_helmet
     val/helmet,   val/no_helmet
4) Trains ConvNeXtV2 Tiny (timm) classifier
5) Produces plots:
   - PR curve
   - Precision & Recall vs confidence threshold
   - Confusion matrix (threshold=0.5 and best-F1 threshold)

Assumptions:
- Image blobs are in containers, label is a .txt with the same basename in the same container.
- YOLO label format per line: cls x y w h (normalized to image size)
- Classes: 0=person, 1=head_helmet, 2=head_nohelmet

Requirements:
  pip install azure-storage-blob opencv-python numpy pandas tqdm torch torchvision timm scikit-learn matplotlib

Notes:
- This uses GT boxes to crop. It tests "Can a classifier separate helmet/no-helmet given a head crop?"
- Keep the same camera-level split you already created (train containers vs val containers).
"""

from __future__ import annotations

import os
import io
import re
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from tqdm import tqdm

from azure.storage.blob import BlobServiceClient

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT THESE)
# =========================
AZURE_CONNECTION_STRING = "<<<PUT_YOUR_CONNECTION_STRING_HERE>>>"

# Put YOUR container names here (full names)
TRAIN_CONTAINERS = [
    # "spais-zst-ziz-0098-train",
]
VAL_CONTAINERS = [
    # "spais-zst-ziz-0020-val",
]

OUT_DIR = "./helmet_cls_crops"     # where crops + outputs go
CROP_PADDING = 0.25               # 25% padding around GT head box
MIN_CROP_SIZE_PX = 8              # skip too tiny crops
MAX_SAMPLES_PER_CLASS_TRAIN = None  # e.g. 20000 for speed, or None for all
MAX_SAMPLES_PER_CLASS_VAL = None

# Training
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If your dataset is very imbalanced, this helps:
USE_WEIGHTED_SAMPLER = True


# =========================
# Utils
# =========================
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_image_blob(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)

def label_name_for_image(img_name: str) -> str:
    base, _ = os.path.splitext(img_name)
    return base + ".txt"

def parse_yolo_labels(txt: str) -> List[Tuple[int, float, float, float, float]]:
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            out.append((cls, x, y, w, h))
        except Exception:
            continue
    return out

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def imdecode_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def crop_with_padding(img: np.ndarray, x: float, y: float, w: float, h: float, pad: float) -> Optional[np.ndarray]:
    """
    x,y,w,h are normalized YOLO (center-x, center-y, width, height)
    """
    H, W = img.shape[:2]
    cx = x * W
    cy = y * H
    bw = w * W
    bh = h * H

    # padding in pixels relative to bbox size
    pad_w = bw * pad
    pad_h = bh * pad

    x1 = int(round(cx - bw / 2 - pad_w))
    y1 = int(round(cy - bh / 2 - pad_h))
    x2 = int(round(cx + bw / 2 + pad_w))
    y2 = int(round(cy + bh / 2 + pad_h))

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < MIN_CROP_SIZE_PX or crop.shape[1] < MIN_CROP_SIZE_PX:
        return None
    return crop

def save_jpg(path: str, img_bgr: np.ndarray, quality: int = 95):
    ensure_dir(os.path.dirname(path))
    cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tofile(path)


# =========================
# Azure: download + crop
# =========================
@dataclass
class Stats:
    helmet: int = 0
    no_helmet: int = 0
    skipped_tiny: int = 0
    skipped_no_label: int = 0
    skipped_no_img: int = 0

def download_and_make_crops(
    service: BlobServiceClient,
    containers: List[str],
    split_name: str,
    out_dir: str,
    max_per_class: Optional[int] = None,
) -> Stats:
    """
    Creates crops for given split: train or val.
    """
    stats = Stats()
    helmet_dir = os.path.join(out_dir, split_name, "helmet")
    nohelmet_dir = os.path.join(out_dir, split_name, "no_helmet")
    ensure_dir(helmet_dir); ensure_dir(nohelmet_dir)

    # counters to limit per class if needed
    saved_helmet = 0
    saved_nohelmet = 0

    for cont in tqdm(containers, desc=f"[{split_name}] containers"):
        cc = service.get_container_client(cont)

        # list all blobs once, build a set for existence checks
        # (faster than calling exists() for each label)
        blob_names = []
        for b in cc.list_blobs():
            blob_names.append(b.name)
        blob_set = set(blob_names)

        # image blobs
        img_blobs = [n for n in blob_names if is_image_blob(n)]

        for img_name in tqdm(img_blobs, desc=f"[{split_name}] {cont}", leave=False):
            if max_per_class is not None and saved_helmet >= max_per_class and saved_nohelmet >= max_per_class:
                break

            lbl_name = label_name_for_image(img_name)
            if lbl_name not in blob_set:
                stats.skipped_no_label += 1
                continue

            # download image
            try:
                img_bytes = cc.get_blob_client(img_name).download_blob().readall()
                img = imdecode_bytes(img_bytes)
                if img is None:
                    stats.skipped_no_img += 1
                    continue
            except Exception:
                stats.skipped_no_img += 1
                continue

            # download label
            try:
                lbl_bytes = cc.get_blob_client(lbl_name).download_blob().readall()
                lbl_txt = lbl_bytes.decode("utf-8", errors="ignore")
            except Exception:
                stats.skipped_no_label += 1
                continue

            labels = parse_yolo_labels(lbl_txt)
            if not labels:
                continue

            # produce crops for classes 1 and 2 only
            # class 1 -> helmet, class 2 -> no_helmet
            base = os.path.splitext(os.path.basename(img_name))[0]
            # include container in filename for uniqueness
            cont_tag = re.sub(r"[^0-9A-Za-z]+", "_", cont)

            head_idx = 0
            for cls, x, y, w, h in labels:
                if cls not in (1, 2):
                    continue

                crop = crop_with_padding(img, x, y, w, h, CROP_PADDING)
                if crop is None:
                    stats.skipped_tiny += 1
                    continue

                head_idx += 1
                if cls == 1:
                    if max_per_class is not None and saved_helmet >= max_per_class:
                        continue
                    out_path = os.path.join(helmet_dir, f"{cont_tag}__{base}__h{head_idx}.jpg")
                    save_jpg(out_path, crop)
                    saved_helmet += 1
                    stats.helmet += 1
                else:
                    if max_per_class is not None and saved_nohelmet >= max_per_class:
                        continue
                    out_path = os.path.join(nohelmet_dir, f"{cont_tag}__{base}__h{head_idx}.jpg")
                    save_jpg(out_path, crop)
                    saved_nohelmet += 1
                    stats.no_helmet += 1

    return stats


# =========================
# Training + evaluation
# =========================
def build_dataloaders(data_root: str):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    tfm_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm_val)

    # class mapping: folder name sorted alphabetically
    # Expect: helmet, no_helmet
    print("Class to index:", train_ds.class_to_idx)

    if USE_WEIGHTED_SAMPLER:
        # compute weights to balance classes
        targets = np.array([y for _, y in train_ds.samples], dtype=int)
        class_counts = np.bincount(targets)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        shuffle = False
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        shuffle = True

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_ds, val_ds, train_loader, val_loader

def build_model(num_classes: int):
    # ConvNeXtV2 Tiny
    model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=num_classes)
    return model

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    all_probs = []
    all_targets = []

    for xb, yb in tqdm(loader, desc="Eval", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return probs, targets

def train_one_run(data_root: str, out_dir: str):
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(data_root)

    num_classes = len(train_ds.classes)
    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_ap = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")

    history = {"epoch": [], "train_loss": [], "val_ap": [], "val_prec@0.5": [], "val_rec@0.5": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=running / max(n, 1))

        scheduler.step()

        # evaluate AP on val (for class "no_helmet" by default)
        probs, targets = eval_model(model, val_loader)

        # define positive class as "no_helmet"
        # get index from class_to_idx (val_ds shares same mapping)
        idx_nohelmet = val_ds.class_to_idx.get("no_helmet", 1)
        y_true = (targets == idx_nohelmet).astype(int)
        y_score = probs[:, idx_nohelmet]

        ap = average_precision_score(y_true, y_score)

        # quick P/R at threshold 0.5
        y_pred_05 = (y_score >= 0.5).astype(int)
        prec_05 = precision_score(y_true, y_pred_05, zero_division=0)
        rec_05 = recall_score(y_true, y_pred_05, zero_division=0)

        history["epoch"].append(epoch)
        history["train_loss"].append(running / max(n, 1))
        history["val_ap"].append(ap)
        history["val_prec@0.5"].append(prec_05)
        history["val_rec@0.5"].append(rec_05)

        print(f"Epoch {epoch}: train_loss={history['train_loss'][-1]:.4f}  val_AP={ap:.4f}  P@0.5={prec_05:.3f} R@0.5={rec_05:.3f}")

        if ap > best_ap:
            best_ap = ap
            torch.save({"model": model.state_dict(), "class_to_idx": val_ds.class_to_idx}, best_path)

    # save training curve
    hist_path = os.path.join(out_dir, "train_history.csv")
    import pandas as pd
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print("Saved history:", hist_path)
    print("Saved best model:", best_path)

    # reload best and make final plots
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs, targets = eval_model(model, val_loader)
    plot_all(val_ds.class_to_idx, probs, targets, out_dir)

def plot_all(class_to_idx: Dict[str, int], probs: np.ndarray, targets: np.ndarray, out_dir: str):
    ensure_dir(out_dir)
    idx_nohelmet = class_to_idx.get("no_helmet", 1)
    y_true = (targets == idx_nohelmet).astype(int)
    y_score = probs[:, idx_nohelmet]

    # PR curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AP={ap:.4f})")
    pr_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", pr_path)

    # Precision/Recall vs threshold
    # sklearn returns thr length = len(prec)-1; align by using prec[1:], rec[1:]
    if len(thr) > 0:
        p_thr = prec[1:]
        r_thr = rec[1:]
        plt.figure()
        plt.plot(thr, p_thr, label="Precision")
        plt.plot(thr, r_thr, label="Recall")
        plt.xlabel("Threshold (no_helmet confidence)")
        plt.ylabel("Score")
        plt.title("Precision/Recall vs Threshold")
        plt.legend()
        prt_path = os.path.join(out_dir, "p_r_vs_threshold.png")
        plt.savefig(prt_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", prt_path)

        # choose best F1 threshold
        f1 = 2 * p_thr * r_thr / np.maximum(p_thr + r_thr, 1e-12)
        best_i = int(np.nanargmax(f1))
        best_thr = float(thr[best_i])
    else:
        best_thr = 0.5

    # Confusion matrices
    def save_cm(threshold: float, name: str):
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0=helmet, 1=no_helmet

        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix (thr={threshold:.3f})")
        plt.xticks([0, 1], ["helmet", "no_helmet"])
        plt.yticks([0, 1], ["helmet", "no_helmet"])
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # numbers
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        cm_path = os.path.join(out_dir, name)
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", cm_path)

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        print(f"[thr={threshold:.3f}] Precision={p:.3f} Recall={r:.3f}")

    save_cm(0.5, "confusion_matrix_thr_0p5.png")
    save_cm(best_thr, "confusion_matrix_best_f1.png")

    # Save numeric summary
    summary_path = os.path.join(out_dir, "val_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"AP={ap:.6f}\n")
        f.write(f"best_f1_threshold={best_thr:.6f}\n")
    print("Saved:", summary_path)


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    outputs_dir = os.path.join(OUT_DIR, "outputs")
    ensure_dir(outputs_dir)

    # 1) Azure connect
    service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    # 2) Make crops
    print("Creating TRAIN crops...")
    tr_stats = download_and_make_crops(
        service, TRAIN_CONTAINERS, "train", OUT_DIR, max_per_class=MAX_SAMPLES_PER_CLASS_TRAIN
    )
    print("TRAIN stats:", tr_stats)

    print("\nCreating VAL crops...")
    va_stats = download_and_make_crops(
        service, VAL_CONTAINERS, "val", OUT_DIR, max_per_class=MAX_SAMPLES_PER_CLASS_VAL
    )
    print("VAL stats:", va_stats)

    # 3) Train classifier
    print(f"\nTraining ConvNeXtV2 Tiny on crops. DEVICE={DEVICE}")
    train_one_run(OUT_DIR, outputs_dir)

    print("\nDONE.")
    print("Check outputs in:", outputs_dir)


if __name__ == "__main__":
    main()