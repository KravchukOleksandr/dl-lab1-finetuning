# grad_accum_single_images.py
import os, math, random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms as T
import timm

# ---------- ПАРАМЕТРЫ ----------
DATA_ROOT  = "path/to/DATA_ROOT"        # hallu/ , person/
CLASSES    = ["hallu", "person"]
MODEL_ID   = "convnextv2_tiny.fcmae_ft_in22k_in1k"

MAX_LONG   = 192
MIN_SHORT  = 32
EPOCHS     = 3
LR         = 1e-4
WD         = 5e-2
ACCUM_STEPS = 32  # накопление градиентов (эквивалент batch=32)

OUT_WEIGHTS = f"{MODEL_ID.replace('.', '_')}_gradaccum.pth"

# ---------- ЗАГРУЗКА ИЗОБРАЖЕНИЯ ----------
def load_image_rect_safe(path: str, max_long: int = MAX_LONG, min_short: int = MIN_SHORT, normalize: bool = True):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s_down = max_long / max(w, h)
    s_up   = min_short / min(w, h)
    s = max(s_up, s_down)
    new_w = int(math.ceil(w * s))
    new_h = int(math.ceil(h * s))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    tfms = [T.ToTensor()]
    if normalize:
        tfms.append(T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)))
    return T.Compose(tfms)(img)  # [3,H,W]

# ---------- ДАТАСЕТ ----------
def collect_files_balanced(root: str):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    per_class = []
    for cname in CLASSES:
        files = sorted(str(p) for p in (Path(root)/cname).rglob("*") if p.suffix.lower() in exts)
        per_class.append(files)
    sizes = [len(x) for x in per_class]
    i_min = 0 if sizes[0] <= sizes[1] else 1
    i_max = 1 - i_min
    minor = per_class[i_min]
    major = random.sample(per_class[i_max], len(minor))
    files = minor + major
    labels = [i_min]*len(minor) + [i_max]*len(major)
    idx = torch.randperm(len(files)).tolist()
    return [files[i] for i in idx], [labels[i] for i in idx], sizes

class FilesDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files; self.labels = labels
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = load_image_rect_safe(self.files[i])
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

# ---------- ПОДГОТОВКА ----------
files, labels, sizes = collect_files_balanced(DATA_ROOT)
ds = FilesDataset(files, labels)

backbone = timm.create_model(MODEL_ID, pretrained=True, num_classes=0).eval().cuda()
feat_dim = backbone.num_features if hasattr(backbone, "num_features") else \
           backbone(torch.zeros(1,3,MIN_SHORT,MIN_SHORT).cuda()).shape[1]
head = nn.Linear(feat_dim, len(CLASSES)).cuda()
model = nn.Sequential(backbone, head).train().cuda()

opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()

# ---------- ОБУЧЕНИЕ С НАКОПЛЕНИЕМ ----------
opt.zero_grad(set_to_none=True)
step = 0

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(ds):
        x = x.unsqueeze(0).cuda()   # [1,3,H,W]
        y = y.unsqueeze(0).cuda()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()             # копим градиенты
        step += 1
        if step % ACCUM_STEPS == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
    # сброс в конце эпохи
    if step % ACCUM_STEPS != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

# ---------- СОХРАНЕНИЕ ----------
torch.save({
    "model_id": MODEL_ID,
    "state_dict": model.state_dict(),
    "head": {"in_features": feat_dim, "num_classes": len(CLASSES), "classes": CLASSES},
}, OUT_WEIGHTS)

print(f"Saved -> {OUT_WEIGHTS}")
print(f"class sizes: {CLASSES[0]}={sizes[0]}  {CLASSES[1]}={sizes[1]}  (balanced={len(ds)//2} each)")