import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient


# =========================
# CONFIG
# =========================

AZURE_CONNECTION_STRING = "PUT_CONNECTION_STRING_HERE"
CONTAINER_NAME = "sam-crops-masks"

OUTPUT_DIR = Path(".")
ASPECT_PLOT_PATH = OUTPUT_DIR / "aspect_ratio_density.png"
DIAGONAL_PLOT_PATH = OUTPUT_DIR / "diagonal_density.png"

META_SUFFIX = "_meta.json"


# =========================
# UTILS
# =========================

def gaussian_kde_manual(values, points=400):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return None, None

    v_min = values.min()
    v_max = values.max()

    if v_min == v_max:
        return None, None

    xs = np.linspace(v_min, v_max, points)

    std = values.std(ddof=1)
    n = len(values)

    # Silverman's rule of thumb
    bandwidth = 1.06 * std * (n ** (-1 / 5))

    if bandwidth <= 1e-9:
        return None, None

    diffs = (xs[:, None] - values[None, :]) / bandwidth
    ys = np.exp(-0.5 * diffs ** 2).sum(axis=1)
    ys /= n * bandwidth * math.sqrt(2 * math.pi)

    return xs, ys


def plot_density(values, title, xlabel, output_path):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=50,
        density=True,
        alpha=0.35,
        edgecolor="black",
        label="Histogram density",
    )

    xs, ys = gaussian_kde_manual(values)
    if xs is not None:
        plt.plot(xs, ys, linewidth=2, label="KDE")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def box_width_height(box):
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width, height


# =========================
# MAIN
# =========================

def main():
    service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)

    meta_blobs = [
        blob.name
        for blob in container.list_blobs()
        if blob.name.endswith(META_SUFFIX)
    ]

    print(f"Found meta files: {len(meta_blobs)}")

    aspect_ratios = []
    diagonals = []

    for blob_name in tqdm(meta_blobs, desc="Reading meta"):
        data = container.get_blob_client(blob_name).download_blob().readall()
        meta = json.loads(data.decode("utf-8"))

        if "work_box_xyxy" not in meta:
            continue

        box = meta["work_box_xyxy"]
        width, height = box_width_height(box)

        if width <= 0 or height <= 0:
            continue

        aspect_ratios.append(width / height)
        diagonals.append(math.sqrt(width * width + height * height))

    aspect_ratios = np.asarray(aspect_ratios, dtype=np.float64)
    diagonals = np.asarray(diagonals, dtype=np.float64)

    print(f"Valid work boxes: {len(aspect_ratios)}")

    if len(aspect_ratios) == 0:
        print("No valid work_box_xyxy values found.")
        return

    print()
    print("Aspect ratio width / height:")
    print(f"  min:    {aspect_ratios.min():.4f}")
    print(f"  mean:   {aspect_ratios.mean():.4f}")
    print(f"  median: {np.median(aspect_ratios):.4f}")
    print(f"  max:    {aspect_ratios.max():.4f}")

    print()
    print("Diagonal size, px:")
    print(f"  min:    {diagonals.min():.2f}")
    print(f"  mean:   {diagonals.mean():.2f}")
    print(f"  median: {np.median(diagonals):.2f}")
    print(f"  max:    {diagonals.max():.2f}")

    plot_density(
        aspect_ratios,
        title="Aspect Ratio Density, work_box width / height",
        xlabel="width / height",
        output_path=ASPECT_PLOT_PATH,
    )

    plot_density(
        diagonals,
        title="Work Box Diagonal Size Density",
        xlabel="diagonal size, px",
        output_path=DIAGONAL_PLOT_PATH,
    )

    print()
    print(f"Saved: {ASPECT_PLOT_PATH.resolve()}")
    print(f"Saved: {DIAGONAL_PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()