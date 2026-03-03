"""
Build scenario_passport.csv and camera_passport.csv from Azure Blob Storage containers.

Assumptions (per your message):
- You provide AZURE_CONNECTION_STRING at the top.
- We process ALL containers whose name starts with: spais-zst-ziz-<camera_name>
  *camera_name* is everything after the prefix.
- All containers with the same <camera_name> belong to ONE camera (camera_id).
- session_id for a camera is the index of a "day", where the day is defined by
  blob.last_modified.date() (UTC) for the IMAGE blob.
  All frames with the same last_modified DAY belong to the same session.
- Each frame is an image file, and its label is a .txt file with the SAME basename
  in the SAME container.
- YOLO label format per line: cls x y w h  (normalized)
- Classes: 0=person, 1=head_helmet, 2=head_nohelmet
- No motion/dynamics. No head_to_person ratio needed.

Outputs:
- scenario_passport.csv  (one row = camera_id + session_id)
- camera_passport.csv    (aggregated per camera_id)

Progress:
- Uses tqdm for cameras/sessions and label parsing progress.

Dependencies:
  pip install azure-storage-blob opencv-python numpy pandas tqdm
"""

from __future__ import annotations

import os
import re
import io
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2  # opencv-python
from azure.storage.blob import BlobServiceClient


# =========================
# CONFIG (edit these)
# =========================
AZURE_CONNECTION_STRING = "<<<PUT_YOUR_CONNECTION_STRING_HERE>>>"

CONTAINER_PREFIX = "spais-zst-ziz-"

# Which image extensions to treat as frames:
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# How many frames per scenario to sample for image-domain metrics:
MAX_SAMPLE_FRAMES_PER_SCENARIO = 300

# Seed for reproducible sampling:
RNG_SEED = 42

# Output paths:
OUT_SCENARIO_CSV = "scenario_passport.csv"
OUT_CAMERA_CSV = "camera_passport.csv"


# =========================
# Helpers
# =========================
def is_image_blob(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(IMAGE_EXTS)


def corresponding_label_name(image_name: str) -> str:
    # same folder, same stem, .txt
    base, _ = os.path.splitext(image_name)
    return base + ".txt"


def safe_parse_yolo_txt(text: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of (cls, x, y, w, h) normalized.
    Skips malformed lines.
    """
    out = []
    for line in text.splitlines():
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


def compute_frame_metrics_bgr(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Computes per-frame metrics (single value per frame):
    - v_mean: mean(V) from HSV
    - v_std:  std(V)
    - s_mean: mean(S)
    - blur:   variance of Laplacian on grayscale
    - haze:   simple proxy = v_mean / (v_std + eps)
    """
    if img_bgr is None or img_bgr.size == 0:
        return {}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    s_mean = float(np.mean(s))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur = float(lap.var())

    eps = 1e-6
    haze = float(v_mean / (v_std + eps))

    return {
        "v_mean": v_mean,
        "v_std": v_std,
        "s_mean": s_mean,
        "blur": blur,
        "haze": haze,
    }


def percentiles(values: List[float], ps=(15, 50, 85)) -> Dict[int, float]:
    if not values:
        return {p: float("nan") for p in ps}
    arr = np.asarray(values, dtype=np.float64)
    out = np.percentile(arr, ps).tolist()
    return {p: float(v) for p, v in zip(ps, out)}


@dataclass(frozen=True)
class ImgRef:
    camera_id: str
    container: str
    blob_name: str
    day: date  # last_modified.date()


# =========================
# Main pipeline
# =========================
def main() -> None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    # 1) List containers by prefix and group into cameras
    cameras: Dict[str, List[str]] = {}
    all_containers = list(service.list_containers(name_starts_with=CONTAINER_PREFIX))
    if not all_containers:
        raise RuntimeError(f"No containers found with prefix '{CONTAINER_PREFIX}'")

    for c in all_containers:
        name = c["name"] if isinstance(c, dict) else c.name
        cam = name[len(CONTAINER_PREFIX):]
        if not cam:
            continue
        cameras.setdefault(cam, []).append(name)

    # 2) Build list of all image blob references with their last_modified day
    img_refs: List[ImgRef] = []
    print(f"Found {len(cameras)} cameras (grouped by container name suffix).")

    for cam_id, cont_names in tqdm(cameras.items(), desc="Listing blobs per camera"):
        for cont in cont_names:
            cc = service.get_container_client(cont)
            # list_blobs yields BlobProperties (name, last_modified, etc.)
            for b in cc.list_blobs():
                if not is_image_blob(b.name):
                    continue
                lm_day = b.last_modified.date()  # UTC date
                img_refs.append(ImgRef(camera_id=cam_id, container=cont, blob_name=b.name, day=lm_day))

    if not img_refs:
        raise RuntimeError("No image blobs found across containers.")

    # 3) Group by (camera_id, day) -> scenario, and create session_id by day order per camera
    #    session_id = index of day sorted ascending for each camera
    by_cam_day: Dict[Tuple[str, date], List[ImgRef]] = {}
    cam_days: Dict[str, List[date]] = {}

    for ref in img_refs:
        by_cam_day.setdefault((ref.camera_id, ref.day), []).append(ref)
        cam_days.setdefault(ref.camera_id, []).append(ref.day)

    # unique and sort days per camera
    cam_day_to_session: Dict[Tuple[str, date], int] = {}
    for cam_id, days in cam_days.items():
        uniq = sorted(set(days))
        for idx, d in enumerate(uniq):
            cam_day_to_session[(cam_id, d)] = idx

    # 4) Process each scenario -> compute passport row
    scenario_rows: List[Dict[str, object]] = []

    # Precompute total frames for progress bars
    total_frames = sum(len(v) for v in by_cam_day.values())

    pbar_scenarios = tqdm(list(by_cam_day.items()), desc="Processing scenarios", total=len(by_cam_day))
    pbar_labels = tqdm(total=total_frames, desc="Parsing labels (frames)", leave=False)

    for (cam_id, day), refs in pbar_scenarios:
        session_id = cam_day_to_session[(cam_id, day)]
        # Sort refs by blob_name for determinism
        refs = sorted(refs, key=lambda r: (r.container, r.blob_name))

        # sample frames for image-domain metrics
        if len(refs) <= MAX_SAMPLE_FRAMES_PER_SCENARIO:
            sample_refs = refs
        else:
            sample_refs = random.sample(refs, MAX_SAMPLE_FRAMES_PER_SCENARIO)

        # We'll determine image size from the first successfully downloaded sample image
        img_w = None
        img_h = None

        # Collect per-frame image metrics
        v_means, v_stds, s_means, blurs, hazes = [], [], [], [], []

        # Helper to download image bytes and decode
        def download_image(ref: ImgRef) -> Optional[np.ndarray]:
            bc = service.get_blob_client(container=ref.container, blob=ref.blob_name)
            data = bc.download_blob().readall()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        # Helper to download label text (if exists)
        def download_label_text(ref: ImgRef) -> Optional[str]:
            lbl_name = corresponding_label_name(ref.blob_name)
            bc = service.get_blob_client(container=ref.container, blob=lbl_name)
            try:
                if not bc.exists():
                    return None
                data = bc.download_blob().readall()
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return None

        # 4a) Compute image-domain metrics on samples
        for sref in sample_refs:
            try:
                img = download_image(sref)
            except Exception:
                continue
            if img is None:
                continue
            if img_w is None or img_h is None:
                img_h, img_w = img.shape[:2]
            m = compute_frame_metrics_bgr(img)
            if not m:
                continue
            v_means.append(m["v_mean"])
            v_stds.append(m["v_std"])
            s_means.append(m["s_mean"])
            blurs.append(m["blur"])
            hazes.append(m["haze"])

        # If we never decoded any image, leave dims/metrics as NaN
        if img_w is None or img_h is None:
            img_w = np.nan
            img_h = np.nan

        # 4b) Parse ALL labels to compute box counts and object-scale stats
        n_person_boxes = 0
        n_head_helmet_boxes = 0
        n_head_nohelmet_boxes = 0

        head_h_px = []
        head_area_px = []
        person_h_px = []

        frames_with_person = 0
        frames_with_nohelmet = 0

        for ref in refs:
            txt = download_label_text(ref)
            pbar_labels.update(1)
            if not txt:
                continue

            # If dims unknown (no sample image decoded), attempt to decode this frame once to get dims
            if (isinstance(img_w, float) and np.isnan(img_w)) or (isinstance(img_h, float) and np.isnan(img_h)):
                try:
                    img = download_image(ref)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                except Exception:
                    pass

            labels = safe_parse_yolo_txt(txt)
            if not labels:
                continue

            has_person = False
            has_nohelmet = False

            for cls, x, y, w, h in labels:
                if cls == 0:
                    n_person_boxes += 1
                    has_person = True
                    if not (isinstance(img_h, float) and np.isnan(img_h)):
                        ph = h * float(img_h)
                        person_h_px.append(ph)
                elif cls == 1:
                    n_head_helmet_boxes += 1
                    if not (isinstance(img_h, float) and np.isnan(img_h)):
                        hh = h * float(img_h)
                        hw = w * float(img_w)
                        head_h_px.append(hh)
                        head_area_px.append(hw * hh)
                elif cls == 2:
                    n_head_nohelmet_boxes += 1
                    has_nohelmet = True
                    if not (isinstance(img_h, float) and np.isnan(img_h)):
                        hh = h * float(img_h)
                        hw = w * float(img_w)
                        head_h_px.append(hh)
                        head_area_px.append(hw * hh)

            if has_person:
                frames_with_person += 1
            if has_nohelmet:
                frames_with_nohelmet += 1

        # Percentiles for object scale
        head_h_pct = percentiles(head_h_px, ps=(10, 50, 90))
        head_area_pct = percentiles(head_area_px, ps=(10, 50, 90))
        person_h_pct = percentiles(person_h_px, ps=(50,))  # only p50 requested

        # Percentiles for domain metrics
        v_pct = percentiles(v_means, ps=(15, 50, 85))
        vstd_pct = percentiles(v_stds, ps=(15, 50, 85))
        s_pct = percentiles(s_means, ps=(15, 50, 85))
        blur_pct = percentiles(blurs, ps=(15, 50, 85))
        haze_pct = percentiles(hazes, ps=(15, 50, 85))

        row = {
            # IDs
            "camera_id": cam_id,
            "session_id": session_id,
            "session_day_utc": str(day),
            "n_frames": len(refs),
            "img_w": int(img_w) if not (isinstance(img_w, float) and np.isnan(img_w)) else np.nan,
            "img_h": int(img_h) if not (isinstance(img_h, float) and np.isnan(img_h)) else np.nan,

            # GT counts
            "n_person_boxes": n_person_boxes,
            "n_head_helmet_boxes": n_head_helmet_boxes,
            "n_head_nohelmet_boxes": n_head_nohelmet_boxes,
            "frames_with_person": frames_with_person,
            "frames_with_nohelmet": frames_with_nohelmet,

            # Object scale percentiles
            "head_h_p10": head_h_pct[10],
            "head_h_p50": head_h_pct[50],
            "head_h_p90": head_h_pct[90],
            "head_area_p10": head_area_pct[10],
            "head_area_p50": head_area_pct[50],
            "head_area_p90": head_area_pct[90],
            "person_h_p50": person_h_pct[50],

            # Domain percentiles (per-frame)
            "v_p15": v_pct[15],
            "v_p50": v_pct[50],
            "v_p85": v_pct[85],

            "v_std_p15": vstd_pct[15],
            "v_std_p50": vstd_pct[50],
            "v_std_p85": vstd_pct[85],

            "s_p15": s_pct[15],
            "s_p50": s_pct[50],
            "s_p85": s_pct[85],

            "blur_p15": blur_pct[15],
            "blur_p50": blur_pct[50],
            "blur_p85": blur_pct[85],

            "haze_p15": haze_pct[15],
            "haze_p50": haze_pct[50],
            "haze_p85": haze_pct[85],
        }
        scenario_rows.append(row)

    pbar_labels.close()

    # 5) Save scenario passport
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.sort_values(["camera_id", "session_id"], inplace=True)
    scenario_df.to_csv(OUT_SCENARIO_CSV, index=False)
    print(f"Saved: {OUT_SCENARIO_CSV}  ({len(scenario_df)} rows)")

    # 6) Build camera passport by aggregating scenarios
    # Totals:
    agg_total = scenario_df.groupby("camera_id", as_index=False).agg(
        n_sessions=("session_id", "nunique"),
        n_frames_total=("n_frames", "sum"),
        n_person_total=("n_person_boxes", "sum"),
        n_head_helmet_total=("n_head_helmet_boxes", "sum"),
        n_head_nohelmet_total=("n_head_nohelmet_boxes", "sum"),
    )

    # Medians of key medians (robust):
    agg_med = scenario_df.groupby("camera_id", as_index=False).agg(
        head_h_p50_med=("head_h_p50", "median"),
        head_h_p10_med=("head_h_p10", "median"),
        v_p50_med=("v_p50", "median"),
        v_p15_med=("v_p15", "median"),
        v_p85_med=("v_p85", "median"),
        blur_p50_med=("blur_p50", "median"),
        haze_p50_med=("haze_p50", "median"),
    )

    camera_df = agg_total.merge(agg_med, on="camera_id", how="left")
    camera_df.sort_values(["n_head_nohelmet_total", "n_frames_total"], ascending=False, inplace=True)
    camera_df.to_csv(OUT_CAMERA_CSV, index=False)
    print(f"Saved: {OUT_CAMERA_CSV}  ({len(camera_df)} cameras)")

    print("\nDone.")


if __name__ == "__main__":
    main()
