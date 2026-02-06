# reference_data.py
from dataclasses import dataclass
from typing import Any


@dataclass
class ReferenceData:
    ref_frame: Any                # np.ndarray (BGR)
    ref_zone_2d: Any              # 2D зона
    ref_plane_indices: Any        # индексы плоскостей по точкам зоны
    ref_planes: Any               # плоскости
    ref_features: Any             # superpoint features


# reference_loader.py
from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import numpy as np
import cv2

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from reference_data import ReferenceData


# =========================
# Blob utils
# =========================

def _download_blob_bytes(
    bsc: BlobServiceClient,
    container: str,
    blob_name: str,
) -> bytes:
    blob = bsc.get_container_client(container).get_blob_client(blob_name)
    return blob.download_blob().readall()


# =========================
# Decode / parse
# =========================

def _decode_png(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode reference image.png")
    return img


def _load_json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


# =========================
# Domain hooks (твои функции)
# =========================

def split_zone3d(
    zone3d_json: Any,
) -> tuple[Any, Any]:
    """
    Разбивает 3D-зону на:
      - 2D-зону
      - индексы плоскостей для каждой точки

    TODO: подключить реальную реализацию
    """
    raise NotImplementedError


def extract_superpoint_features(
    superpoint_model: Any,
    bgr_image: np.ndarray,
) -> Any:
    """
    TODO: подключить реальный экстракт фич
    """
    raise NotImplementedError


# =========================
# Public API
# =========================

def load_reference_data_from_blob(
    *,
    storage_account_url: str,
    container_name: str,
    prefix: str,
    superpoint_model: Any,
) -> ReferenceData:
    """
    Загружает и подготавливает опорные данные.

    Ожидаемые файлы в blob:
      {prefix}/image.png
      {prefix}/zone3d.json
      {prefix}/planes.json
    """
    credential = DefaultAzureCredential()
    bsc = BlobServiceClient(
        account_url=storage_account_url,
        credential=credential,
    )

    # --- download ---
    img_bytes = _download_blob_bytes(
        bsc, container_name, f"{prefix}image.png"
    )
    zone3d_bytes = _download_blob_bytes(
        bsc, container_name, f"{prefix}zone3d.json"
    )
    planes_bytes = _download_blob_bytes(
        bsc, container_name, f"{prefix}planes.json"
    )

    # --- parse ---
    ref_frame = _decode_png(img_bytes)
    zone3d = _load_json(zone3d_bytes)
    ref_planes = _load_json(planes_bytes)

    # --- domain transforms ---
    ref_zone_2d, ref_plane_indices = split_zone3d(zone3d)
    ref_features = extract_superpoint_features(
        superpoint_model,
        ref_frame,
    )

    return ReferenceData(
        ref_frame=ref_frame,
        ref_zone_2d=ref_zone_2d,
        ref_plane_indices=ref_plane_indices,
        ref_planes=ref_planes,
        ref_features=ref_features,
    )