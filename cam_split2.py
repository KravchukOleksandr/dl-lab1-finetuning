import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

K = 5  # количество фолдов

df = df.copy()

# --- 1. биннинг по масштабу головы
df["head_bin"] = pd.qcut(
    df["head_h_p50_med"],
    q=3,
    labels=["small", "medium", "large"]
)

# --- 2. биннинг по количеству no-helmet
df["noh_bin"] = pd.qcut(
    df["n_head_nohelmet_total"],
    q=3,
    labels=["low", "mid", "high"]
)

# --- 3. комбинированная страта
df["strata"] = df["head_bin"].astype(str) + "_" + df["noh_bin"].astype(str)

X = df["camera_id"].values
y = df["strata"].values

skf = StratifiedKFold(
    n_splits=K,
    shuffle=True,
    random_state=42
)

folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    train_cameras = df.iloc[train_idx]["camera_id"].tolist()
    val_cameras = df.iloc[val_idx]["camera_id"].tolist()

    print(f"\nFOLD {fold}")
    print("VAL cameras:", val_cameras)
    print("VAL nohelmet:", df.iloc[val_idx]["n_head_nohelmet_total"].sum())
    print("TRAIN cameras:", len(train_cameras))

    folds.append({
        "fold": fold,
        "train_cameras": train_cameras,
        "val_cameras": val_cameras
    })

# folds — это список всех разбиений