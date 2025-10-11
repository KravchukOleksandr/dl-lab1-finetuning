# train_convnextv2_tiny_balanced_val.py
import os, math, random
from pathlib import Path
from typing import List, Iterator
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
import timm

# ---------- CONFIG ----------
DATA_ROOT   = "path/to/DATA_ROOT"   # DATA_ROOT/hallu/* , DATA_ROOT/person/*
CLASSES     = ["hallu", "person"]
MODEL_ID    = "convnextv2_tiny.fcmae_ft_in22k_in1k"

MAX_LONG    = 192
MIN_SHORT   = 32
EPOCHS      = 3
ACCUM_STEPS = 32
LR          = 1e-4
WD          = 5e-2
NUM_WORKERS = 2
VAL_SPLIT   = 0.1  # доля валидации

OUT_WEIGHTS = f"{MODEL_ID.replace('.', '_')}_balanced_val.pth"

# ---------- IMAGE LOADING ----------
def load_image_rect_safe(path: str, max_long: int = MAX_LONG, min_short: int = MIN_SHORT):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s_down = max_long / max(w, h)
    s_up   = min_short / min(w, h)
    s = max(s_up, s_down)
    new_w = int(math.ceil(w * s))
    new_h = int(math.ceil(h * s))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    x = T.ToTensor()(img)
    x = T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))(x)
    return x

# ---------- DATASET ----------
class ImageFolderRectDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = load_image_rect_safe(self.files[i])
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

# ---------- SAMPLER ----------
class BalancedPerEpochSampler(Sampler[int]):
    def __init__(self, labels: List[int]):
        self.labels = labels
        self.idx_by_cls = {0: [], 1: []}
        for i, y in enumerate(labels):
            self.idx_by_cls[int(y)].append(i)
        self.n0, self.n1 = len(self.idx_by_cls[0]), len(self.idx_by_cls[1])
    def __iter__(self):
        if self.n0 <= self.n1:
            minor_cls, major_cls = 0, 1
        else:
            minor_cls, major_cls = 1, 0
        minor_idx = self.idx_by_cls[minor_cls][:]
        major_idx = random.sample(self.idx_by_cls[major_cls], len(minor_idx))
        idx = minor_idx + major_idx
        random.shuffle(idx)
        return iter(idx)
    def __len__(self): return 2 * min(self.n0, self.n1)

def collate_single_to_cuda(batch):
    x, y = batch[0]
    return x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda()

# ---------- SPLIT TRAIN/VAL ----------
def collect_all_files(root: str):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    files, labels = [], []
    for ci, cname in enumerate(CLASSES):
        for p in Path(root, cname).rglob("*"):
            if p.suffix.lower() in exts:
                files.append(str(p))
                labels.append(ci)
    return files, labels

files, labels = collect_all_files(DATA_ROOT)
idx = list(range(len(files)))
random.shuffle(idx)
split = int(len(files)*(1-VAL_SPLIT))
train_idx, val_idx = idx[:split], idx[split:]

train_files  = [files[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_files    = [files[i] for i in val_idx]
val_labels   = [labels[i] for i in val_idx]

train_ds = ImageFolderRectDataset(train_files, train_labels)
val_ds   = ImageFolderRectDataset(val_files, val_labels)

train_sampler = BalancedPerEpochSampler(train_ds.labels)
train_loader  = DataLoader(train_ds, batch_size=1, sampler=train_sampler,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           collate_fn=collate_single_to_cuda)
val_loader    = DataLoader(val_ds, batch_size=1, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           collate_fn=collate_single_to_cuda)

# ---------- MODEL ----------
backbone = timm.create_model(MODEL_ID, pretrained=True, num_classes=0).eval().cuda()
feat_dim = backbone.num_features if hasattr(backbone, "num_features") else \
           backbone(torch.zeros(1,3,MIN_SHORT,MIN_SHORT).cuda()).shape[1]
head = nn.Linear(feat_dim, len(CLASSES)).cuda()
model = nn.Sequential(backbone, head).train().cuda()

opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()

# ---------- TRAIN + VALIDATION ----------
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad(set_to_none=True)
    step, running_loss, correct, total = 0, 0.0, 0, 0

    for i, (x, y) in enumerate(train_loader):
        out = model(x)
        loss = crit(out, y) / ACCUM_STEPS
        loss.backward()
        running_loss += loss.item() * ACCUM_STEPS
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        step += 1
        if step % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            opt.step(); opt.zero_grad(set_to_none=True)

    train_loss = running_loss / total
    train_acc  = correct / total

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            loss = crit(out, y)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)
    val_loss /= val_total
    val_acc  = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"train loss: {train_loss:.4f} acc: {train_acc*100:.1f}% | "
          f"val loss: {val_loss:.4f} acc: {val_acc*100:.1f}%")

# ---------- SAVE ----------
torch.save({
    "model_id": MODEL_ID,
    "state_dict": model.state_dict(),
    "head": {"in_features": feat_dim, "num_classes": len(CLASSES), "classes": CLASSES},
}, OUT_WEIGHTS)

print(f"\nSaved -> {OUT_WEIGHTS}")