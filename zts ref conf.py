# camera_config_loader.py

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List

import pandas as pd
from azure.storage.blob import BlobServiceClient


# =========================
# 1) Доменная модель
# =========================

@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    camera_name: str
    url: str
    codec: str
    thres: float
    only_daytime: bool
    team_id: str
    channel_id: str


# =========================
# 2) "Хардкод" в одном месте
# =========================

# Имена контейнера и файлов (по умолчанию — как ты описал, что они стабильны)
DEFAULT_CONTAINER_NAME = "YOUR_CONTAINER_NAME"
DEFAULT_EXCEL_A_BLOB_NAME = "A.xlsx"
DEFAULT_EXCEL_B_BLOB_NAME = "B.xlsx"

# Ключи
A_KEY_COL = "compose_project_name"
B_KEY_COL = "camera_id"

# Где брать codec
A_CODEC_COL = "codec"

# Маппинг колонок из таблицы B -> поля CameraConfig (кроме codec)
# !!! Впиши тут реальные названия колонок в таблице B
B_COL = {
    "camera_name": "camera_name",      # например "name" или "camera_title"
    "url": "url",                      # если url уже есть в B; если нет — позже добавишь resolver
    "thres": "thres",                  # например "threshold"
    "only_daytime": "only_daytime",    # например "day_only"
    "team_id": "team_id",              # например "teams_team_id"
    "channel_id": "channel_id",        # например "teams_channel_id"
}


# =========================
# 3) Утилиты чтения Blob -> DataFrame
# =========================

def _download_blob_to_bytes(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_name: str,
) -> bytes:
    container = blob_service_client.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)
    return blob.download_blob().readall()


def _read_excel_from_bytes(
    data: bytes,
    *,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    # pandas сам использует openpyxl для .xlsx
    return pd.read_excel(BytesIO(data), sheet_name=sheet_name)


# =========================
# 4) Валидация входных таблиц (минимальная)
# =========================

def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")


# =========================
# 5) Основная функция загрузки и сборки CameraConfig
# =========================

def load_camera_configs_from_blob(
    *,
    connection_string: str,
    container_name: str = DEFAULT_CONTAINER_NAME,
    excel_a_blob_name: str = DEFAULT_EXCEL_A_BLOB_NAME,
    excel_b_blob_name: str = DEFAULT_EXCEL_B_BLOB_NAME,
    sheet_a: str | int | None = 0,
    sheet_b: str | int | None = 0,
) -> List[CameraConfig]:
    """
    Загружает 2 Excel из одного blob container и собирает список CameraConfig.

    Склейка:
      A.compose_project_name  <->  B.camera_id

    Правила полей:
      - camera_id: берём из B.camera_id (внешний ключ)
      - codec: берём из A.codec
      - остальные поля: берём из таблицы B через B_COL mapping
    """
    bsc = BlobServiceClient.from_connection_string(connection_string)

    # 1) скачать оба файла
    a_bytes = _download_blob_to_bytes(bsc, container_name, excel_a_blob_name)
    b_bytes = _download_blob_to_bytes(bsc, container_name, excel_b_blob_name)

    # 2) прочитать
    df_a = _read_excel_from_bytes(a_bytes, sheet_name=sheet_a)
    df_b = _read_excel_from_bytes(b_bytes, sheet_name=sheet_b)

    # 3) минимальная валидация
    _require_columns(df_a, [A_KEY_COL, A_CODEC_COL], "Table A")
    _require_columns(df_b, [B_KEY_COL] + list(B_COL.values()), "Table B")

    # 4) оставляем в A только ключ и codec (чтобы не тащить лишнее)
    df_a_small = df_a[[A_KEY_COL, A_CODEC_COL]].copy()

    # 5) merge: B left join A (все камеры из B должны остаться)
    merged = df_b.merge(
        df_a_small,
        how="left",
        left_on=B_KEY_COL,
        right_on=A_KEY_COL,
        suffixes=("", "_a"),
    )

    # 6) проверка, что codec нашёлся (если нужно строго — включи строгий режим)
    # Если codec может отсутствовать — убери эту проверку или сделай default.
    if merged[A_CODEC_COL].isna().any():
        missing_ids = merged.loc[merged[A_CODEC_COL].isna(), B_KEY_COL].astype(str).tolist()
        raise ValueError(
            f"Codec is missing after merge for camera_ids (from B) not found in A: {missing_ids[:20]}"
            + (" ..." if len(missing_ids) > 20 else "")
        )

    # 7) собрать CameraConfig
    configs: List[CameraConfig] = []
    for _, row in merged.iterrows():
        cfg = CameraConfig(
            camera_id=str(row[B_KEY_COL]),
            camera_name=str(row[B_COL["camera_name"]]),
            url=str(row[B_COL["url"]]),
            codec=str(row[A_CODEC_COL]),  # берём codec из A
            thres=float(row[B_COL["thres"]]),
            only_daytime=bool(row[B_COL["only_daytime"]]),
            team_id=str(row[B_COL["team_id"]]),
            channel_id=str(row[B_COL["channel_id"]]),
        )
        configs.append(cfg)

    return configs


# =========================
# 6) Пример запуска как скрипт
# =========================

if __name__ == "__main__":
    import os

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    configs = load_camera_configs_from_blob(
        connection_string=conn_str,
        # container_name=... (если нужно переопределять)
        # excel_a_blob_name=...
        # excel_b_blob_name=...
    )

    print(f"Loaded {len(configs)} camera configs")
    print(configs[0] if configs else "No configs")