"""
Build scenario_passport.csv and camera_passport.csv from Azure Blob Storage containers.

UPDATED grouping rule (per your clarification):
- Container name: spais-zst-ziz-<camera_name>-<anything>
- camera_id is extracted as the FIRST token after prefix up to the next '-' (or full remainder if '-' absent)
  Examples:
    spais-zst-ziz-0098-val   -> camera_id = "0098"
    spais-zst-ziz-0098-train -> camera_id = "0098"

Other assumptions:
- session_id is the index of a "day" (UTC) by blob.last_modified.date() for IMAGE blobs.
- Label is a .txt with the same basename in the same container.
- YOLO txt format: cls x y w h (normalized)
- Classes: 0=person, 1=head_helmet, 2=head_nohelmet
- No motion. No head_to_person ratio.

Outputs:
- scenario_passport.csv (one row = camera_id + session_id)
- camera_passport.csv   (aggregated per camera_id)

Dependencies:
  pip install azure-storage-blob opencv-python numpy pandas tqdm
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from azure.storage.blob import BlobServiceClient


# =========================
# CONFIG (edit these)
# =========================
AZURE_CONNECTION_STRING = "<<<PUT_YOUR_CONNECTION_STRING_HERE>>>"

CONTAINER_PREFIX = "spais-zst-ziz-"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAX_SAMPLE_FRAMES_PER_SCENARIO = 300
RNG_SEED = 42

OUT_SCENARIO_CSV = "scenario_passport.csv"
OUT_CAMERA_CSV = "camera_passport.csv"


# =========================
# Helpers
# =========================
def is_image_blob(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def corresponding_label_name(image_name: str) -> str:
    base, _ = os.path.splitext(image_name)
    return base + ".txt"


def safe_parse_yolo_txt(text: str) -> List[Tuple[int, float, float, float, float]]:
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
    if img_bgr is None or img_bgr.size == 0:
        return {}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    s_mean = float(np.mean(s))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    eps = 1e-6
    haze = float(v_mean / (v_std + eps))

    return {"v_mean": v_mean, "v_std": v_std, "s_mean": s_mean, "blur": blur, "haze": haze}


def percentiles(values: List[float], ps=(15, 50, 85)) -> Dict[int, float]:
    if not values:
        return {p: float("nan") for p in ps}
    arr = np.asarray(values, dtype=np.float64)
    out = np.percentile(arr, ps).tolist()
    return {p: float(v) for p, v in zip(ps, out)}


def extract_camera_id(container_name: str) -> Optional[str]:
    """
    Extract camera_id from container_name:
      spais-zst-ziz-<camera_id>-<suffix...>
    camera_id = first token after prefix, split by '-'.
    """
    if not container_name.startswith(CONTAINER_PREFIX):
        return None
    tail = container_name[len(CONTAINER_PREFIX):]  # e.g. "0098-val"
    if not tail:
        return None
    # camera_id is first segment before '-'
    cam = tail.split("-", 1)[0]
    return cam if cam else None


@dataclass(frozen=True)
class ImgRef:
    camera_id: str
    container: str
    blob_name: str
    day: date  # last_modified.date() in UTC


# =========================
# Main pipeline
# =========================
def main() -> None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    # 1) List containers by prefix and group into cameras via extract_camera_id()
    cameras: Dict[str, List[str]] = {}
    all_containers = list(service.list_containers(name_starts_with=CONTAINER_PREFIX))
    if not all_containers:
        raise RuntimeError(f"No containers found with prefix '{CONTAINER_PREFIX}'")

    for c in all_containers:
        name = c["name"] if isinstance(c, dict) else c.name
        cam = extract_camera_id(name)
        if cam is None:
            continue
        cameras.setdefault(cam, []).append(name)

    print(f"Found {len(cameras)} cameras (grouped by first token after prefix).")

    # 2) Build list of all image blob references with their last_modified day
    img_refs: List[ImgRef] = []
    for cam_id, cont_names in tqdm(cameras.items(), desc="Listing blobs per camera"):
        for cont in cont_names:
            cc = service.get_container_client(cont)
            for b in cc.list_blobs():
                if not is_image_blob(b.name):
                    continue
                lm_day = b.last_modified.date()  # UTC date
                img_refs.append(ImgRef(camera_id=cam_id, container=cont, blob_name=b.name, day=lm_day))

    if not img_refs:
        raise RuntimeError("No image blobs found across containers.")

    # 3) Group by (camera_id, day) -> scenario; session_id = index of day per camera
    by_cam_day: Dict[Tuple[str, date], List[ImgRef]] = {}
    cam_days: Dict[str, List[date]] = {}

    for ref in img_refs:
        by_cam_day.setdefault((ref.camera_id, ref.day), []).append(ref)
        cam_days.setdefault(ref.camera_id, []).append(ref.day)

    cam_day_to_session: Dict[Tuple[str, date], int] = {}
    for cam_id, days in cam_days.items():
        uniq = sorted(set(days))
        for idx, d in enumerate(uniq):
            cam_day_to_session[(cam_id, d)] = idx

    # 4) Process scenarios
    scenario_rows: List[Dict[str, object]] = []

    total_frames = sum(len(v) for v in by_cam_day.values())
    pbar_scenarios = tqdm(list(by_cam_day.items()), desc="Processing scenarios", total=len(by_cam_day))
    pbar_labels = tqdm(total=total_frames, desc="Parsing labels (frames)", leave=False)

    for (cam_id, day), refs in pbar_scenarios:
        session_id = cam_day_to_session[(cam_id, day)]
        refs = sorted(refs, key=lambda r: (r.container, r.blob_name))

        # sample frames for image-domain metrics
        sample_refs = refs if len(refs) <= MAX_SAMPLE_FRAMES_PER_SCENARIO else random.sample(refs, MAX_SAMPLE_FRAMES_PER_SCENARIO)

        img_w = None
        img_h = None

        v_means, v_stds, s_means, blurs, hazes = [], [], [], [], []

        def download_image(ref: ImgRef) -> Optional[np.ndarray]:
            bc = service.get_blob_client(container=ref.container, blob=ref.blob_name)
            data = bc.download_blob().readall()
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

        # 4a) domain metrics
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

        if img_w is None or img_h is None:
            img_w = np.nan
            img_h = np.nan

        # 4b) label stats (all frames)
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

            # If dims unknown, try decode this frame once
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
                        person_h_px.append(h * float(img_h))
                elif cls == 1:
                    n_head_helmet_boxes += 1
                    if not (isinstance(img_h, float) and np.isnan(img_h)):
                        hh = h * float(img_h); hw = w * float(img_w)
                        head_h_px.append(hh)
                        head_area_px.append(hw * hh)
                elif cls == 2:
                    n_head_nohelmet_boxes += 1
                    has_nohelmet = True
                    if not (isinstance(img_h, float) and np.isnan(img_h)):
                        hh = h * float(img_h); hw = w * float(img_w)
                        head_h_px.append(hh)
                        head_area_px.append(hw * hh)

            if has_person:
                frames_with_person += 1
            if has_nohelmet:
                frames_with_nohelmet += 1

        # percentiles
        head_h_pct = percentiles(head_h_px, ps=(10, 50, 90))
        head_area_pct = percentiles(head_area_px, ps=(10, 50, 90))
        person_h_pct = percentiles(person_h_px, ps=(50,))

        v_pct = percentiles(v_means, ps=(15, 50, 85))
        vstd_pct = percentiles(v_stds, ps=(15, 50, 85))
        s_pct = percentiles(s_means, ps=(15, 50, 85))
        blur_pct = percentiles(blurs, ps=(15, 50, 85))
        haze_pct = percentiles(hazes, ps=(15, 50, 85))

        scenario_rows.append({
            "camera_id": cam_id,
            "session_id": session_id,
            "session_day_utc": str(day),
            "n_frames": len(refs),
            "img_w": int(img_w) if not (isinstance(img_w, float) and np.isnan(img_w)) else np.nan,
            "img_h": int(img_h) if not (isinstance(img_h, float) and np.isnan(img_h)) else np.nan,

            "n_person_boxes": n_person_boxes,
            "n_head_helmet_boxes": n_head_helmet_boxes,
            "n_head_nohelmet_boxes": n_head_nohelmet_boxes,
            "frames_with_person": frames_with_person,
            "frames_with_nohelmet": frames_with_nohelmet,

            "head_h_p10": head_h_pct[10],
            "head_h_p50": head_h_pct[50],
            "head_h_p90": head_h_pct[90],
            "head_area_p10": head_area_pct[10],
            "head_area_p50": head_area_pct[50],
            "head_area_p90": head_area_pct[90],
            "person_h_p50": person_h_pct[50],

            "v_p15": v_pct[15], "v_p50": v_pct[50], "v_p85": v_pct[85],
            "v_std_p15": vstd_pct[15], "v_std_p50": vstd_pct[50], "v_std_p85": vstd_pct[85],
            "s_p15": s_pct[15], "s_p50": s_pct[50], "s_p85": s_pct[85],
            "blur_p15": blur_pct[15], "blur_p50": blur_pct[50], "blur_p85": blur_pct[85],
            "haze_p15": haze_pct[15], "haze_p50": haze_pct[50], "haze_p85": haze_pct[85],
        })

    pbar_labels.close()

    scenario_df = pd.DataFrame(scenario_rows).sort_values(["camera_id", "session_id"])
    scenario_df.to_csv(OUT_SCENARIO_CSV, index=False)
    print(f"Saved: {OUT_SCENARIO_CSV} ({len(scenario_df)} rows)")

    # camera-level aggregation
    agg_total = scenario_df.groupby("camera_id", as_index=False).agg(
        n_sessions=("session_id", "nunique"),
        n_frames_total=("n_frames", "sum"),
        n_person_total=("n_person_boxes", "sum"),
        n_head_helmet_total=("n_head_helmet_boxes", "sum"),
        n_head_nohelmet_total=("n_head_nohelmet_boxes", "sum"),
    )
    agg_med = scenario_df.groupby("camera_id", as_index=False).agg(
        head_h_p50_med=("head_h_p50", "median"),
        head_h_p10_med=("head_h_p10", "median"),
        v_p50_med=("v_p50", "median"),
        v_p15_med=("v_p15", "median"),
        v_p85_med=("v_p85", "median"),
        blur_p50_med=("blur_p50", "median"),
        haze_p50_med=("haze_p50", "median"),
    )
    camera_df = agg_total.merge(agg_med, on="camera_id", how="left") \
                        .sort_values(["n_head_nohelmet_total", "n_frames_total"], ascending=False)

    camera_df.to_csv(OUT_CAMERA_CSV, index=False)
    print(f"Saved: {OUT_CAMERA_CSV} ({len(camera_df)} cameras)")
    print("Done.")


if __name__ == "__main__":
    main()
