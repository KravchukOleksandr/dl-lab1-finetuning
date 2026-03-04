import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# df уже существует и содержит колонки:
# camera_id, head_size, light_median, light_span

K = 5          # попробуй 5 или 6
SEED = 42

df2 = df.copy().reset_index(drop=True)

features = ["head_size", "light_median", "light_span"]

# 1) нормализация
X = df2[features].to_numpy(dtype=float)
Xn = StandardScaler().fit_transform(X)

# 2) k-means
km = KMeans(n_clusters=K, random_state=SEED, n_init="auto")
labels = km.fit_predict(Xn)
df2["cluster"] = labels

# 3) выбираем в val по 1 камере из каждого кластера:
#    "самую типичную" = ближайшую к центроиду
val_cams = []
for c in range(K):
    idxs = np.where(labels == c)[0]
    center = km.cluster_centers_[c]
    dists = np.linalg.norm(Xn[idxs] - center, axis=1)
    best_local = idxs[np.argmin(dists)]
    val_cams.append(df2.loc[best_local, "camera_id"])

val_cams = list(dict.fromkeys(val_cams))  # на всякий случай уберём дубли

# 4) train/val
val_df = df2[df2["camera_id"].isin(val_cams)].copy()
train_df = df2[~df2["camera_id"].isin(val_cams)].copy()

# 5) вывод
print("K =", K)
print("VAL cameras:", val_cams)
print("VAL size:", len(val_cams), "cameras")
print("TRAIN size:", train_df["camera_id"].nunique(), "cameras")

print("\nCluster sizes:")
print(df2["cluster"].value_counts().sort_index())

print("\nVAL summary:")
print(val_df[["camera_id", "cluster"] + features].sort_values(["cluster", "camera_id"]))

# (опционально) сохранить списки камер
pd.DataFrame({"camera_id": sorted(val_cams)}).to_csv("val_cameras.csv", index=False)
pd.DataFrame({"camera_id": sorted(train_df["camera_id"].tolist())}).to_csv("train_cameras.csv", index=False)