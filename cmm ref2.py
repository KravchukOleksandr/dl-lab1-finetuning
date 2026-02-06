# reference_loader.py
# Загружает опорные (reference) данные для ОДНОЙ камеры по camera_id:
#   {camera_id}/image.png
#   {camera_id}/zone3d.json
#   {camera_id}/planes.json
#
# Возвращает ReferenceData:
#   ref_frame, ref_zone_2d, ref_plane_indices, ref_planes, ref_features

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import cv2

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# -----------------------------
# Data container
# -----------------------------

@dataclass
class ReferenceData:
    ref_frame: Any            # np.ndarray (BGR)
    ref_zone_2d: Any          # ваша 2D зона
    ref_plane_indices: Any    # индексы плоскостей по точкам зоны
    ref_planes: Any           # плоскости (как распарсили planes.json)
    ref_features: Any         # superpoint features


# -----------------------------
# Domain hooks (подключи свои реализации)
# -----------------------------

def split_zone3d_to_zone2d_and_plane_indices(zone3d_json: Any) -> tuple[Any, Any]:
    """
    Должно вернуть: (zone2d, plane_indices_per_point)
    """
    raise NotImplementedError("Plug your split_zone3d_to_zone2d_and_plane_indices implementation here.")


def extract_superpoint_features(superpoint_model: Any, bgr_image: np.ndarray) -> Any:
    """
    Должно вернуть features в вашем формате.
    """
    raise NotImplementedError("Plug your extract_superpoint_features implementation here.")


# -----------------------------
# Blob helpers
# -----------------------------

def _make_blob_service_client(storage_account_url: str) -> BlobServiceClient:
    # Managed Identity / DefaultAzureCredential (работает в Azure + локально при настройке)
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=storage_account_url, credential=credential)


def _download_blob_bytes(bsc: BlobServiceClient, container: str, blob_name: str) -> bytes:
    blob = bsc.get_container_client(container).get_blob_client(blob_name)
    try:
        return blob.download_blob().readall()
    except Exception as e:
        raise FileNotFoundError(f"Failed to download blob '{container}/{blob_name}': {e}") from e


def _decode_png_to_bgr(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode PNG bytes (image.png).")
    return img


def _load_json_bytes(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


# -----------------------------
# Public API
# -----------------------------

def load_reference_data_from_blob(
    *,
    storage_account_url: str,
    container_name: str,
    camera_id: str,
    superpoint_model: Any,
    prefix_template: str = "{camera_id}/",
) -> ReferenceData:
    """
    Загружает и подготавливает reference data для одной камеры.

    Ожидаемые файлы в blob:
      {prefix}image.png
      {prefix}zone3d.json
      {prefix}planes.json

    prefix по умолчанию = "{camera_id}/"
    """
    camera_id = camera_id.strip()
    if not camera_id:
        raise ValueError("camera_id is empty")

    prefix = prefix_template.format(camera_id=camera_id)
    # на всякий случай: если prefix_template без "/" на конце
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    bsc = _make_blob_service_client(storage_account_url)

    # Download
    img_bytes = _download_blob_bytes(bsc, container_name, f"{prefix}image.png")
    zone3d_bytes = _download_blob_bytes(bsc, container_name, f"{prefix}zone3d.json")
    planes_bytes = _download_blob_bytes(bsc, container_name, f"{prefix}planes.json")

    # Parse
    ref_frame = _decode_png_to_bgr(img_bytes)
    zone3d_json = _load_json_bytes(zone3d_bytes)
    ref_planes = _load_json_bytes(planes_bytes)

    # Transform
    ref_zone_2d, ref_plane_indices = split_zone3d_to_zone2d_and_plane_indices(zone3d_json)
    ref_features = extract_superpoint_features(superpoint_model, ref_frame)

    return ReferenceData(
        ref_frame=ref_frame,
        ref_zone_2d=ref_zone_2d,
        ref_plane_indices=ref_plane_indices,
        ref_planes=ref_planes,
        ref_features=ref_features,
    )