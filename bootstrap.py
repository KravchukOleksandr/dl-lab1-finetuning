# loaders/camera_loader.py
from dataclasses import dataclass
from typing import Any

from loaders.camera_config_loader import load_camera_configs_from_blob, CameraConfig
from loaders.reference_loader import load_reference_data_from_blob, ReferenceData


@dataclass
class Camera:
    config: CameraConfig
    reference: ReferenceData


def load_cameras(
    *,
    storage_account_url: str,
    config_container: str,
    reference_container: str,
    superpoint_model: Any,
    url_base: str | None = None,
) -> list[Camera]:
    configs = load_camera_configs_from_blob(
        storage_account_url=storage_account_url,
        container_name=config_container,
        url_base=url_base,
    )

    cameras: list[Camera] = []
    for cfg in configs:
        ref = load_reference_data_from_blob(
            storage_account_url=storage_account_url,
            container_name=reference_container,
            camera_id=cfg.camera_id,
            superpoint_model=superpoint_model,
        )
        cameras.append(Camera(config=cfg, reference=ref))

    return cameras