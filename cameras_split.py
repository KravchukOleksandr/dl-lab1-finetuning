import pandas as pd
import numpy as np

CAMERA_CSV = "camera_passport.csv"
VAL_SIZE = 5  # можно 6

df = pd.read_csv(CAMERA_CSV)

# оставляем только камеры с no-helmet (вы уже так сделали, но на всякий случай)
df = df[df["n_head_nohelmet_total"] > 0].copy()

# бинning по масштабу головы (квантили)
df["head_bin"] = pd.qcut(df["head_h_p50_med"], q=3, labels=["small", "medium", "large"])

val_ids = set()

# 1) по одной камере из каждого бина: выбираем экстремумы по яркости
for b in ["small", "medium", "large"]:
    g = df[df["head_bin"] == b].copy()
    if len(g) == 0:
        continue
    # берём либо самую тёмную, либо самую светлую — чередуем
    pick = g.sort_values("v_p50_med").iloc[0]["camera_id"]  # самая тёмная
    val_ids.add(pick)

# 2) добираем камеры, чтобы val_size достигнуть VAL_SIZE
# кандидатам отдаём приоритет: высокий haze (дымнее) и низкий blur (более мыльно)
remain = df[~df["camera_id"].isin(val_ids)].copy()
remain["score"] = (
    remain["haze_p50_med"].rank(pct=True)  # больше haze -> выше
    + (1 - remain["blur_p50_med"].rank(pct=True))  # меньше blur -> выше
)
remain = remain.sort_values("score", ascending=False)

for cam in remain["camera_id"].tolist():
    if len(val_ids) >= VAL_SIZE:
        break
    val_ids.add(cam)

val_df = df[df["camera_id"].isin(val_ids)].copy()
train_df = df[~df["camera_id"].isin(val_ids)].copy()

# 3) страховка: если в val слишком мало no-helmet, можно заменить одну камеру
# (порог подберите сами; пример: хотя бы 20% от общего)
total_noh = df["n_head_nohelmet_total"].sum()
val_noh = val_df["n_head_nohelmet_total"].sum()
if val_noh < 0.2 * total_noh:
    # заменим "самую бедную" по no-helmet камеру в val на "самую богатую" из train
    worst_val = val_df.sort_values("n_head_nohelmet_total").iloc[0]["camera_id"]
    best_train = train_df.sort_values("n_head_nohelmet_total", ascending=False).iloc[0]["camera_id"]
    val_ids.remove(worst_val)
    val_ids.add(best_train)

val_df = df[df["camera_id"].isin(val_ids)].copy()
train_df = df[~df["camera_id"].isin(val_ids)].copy()

# сохраняем
val_df[["camera_id"]].to_csv("val_cameras.csv", index=False)
train_df[["camera_id"]].to_csv("train_cameras.csv", index=False)

print("VAL cameras:", sorted(val_ids))
print("VAL no-helmet boxes:", int(val_df["n_head_nohelmet_total"].sum()))
print("TRAIN cameras:", train_df["camera_id"].nunique())