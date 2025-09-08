# -*- coding: utf-8 -*-
# Простое объединение CSV с растяжением времени per-file в общее окно.

import pandas as pd
from typing import List

# ================== НАСТРОЙКИ ==================
FILES: List[str] = [
    "data/file1.csv",
    "data/file2.csv",
    "data/file3.csv",
]
ENTRY_COL = "entry_time"   # UNIX float (секунды)
EXIT_COL  = "exit_time"    # UNIX float (секунды)

# Целевое окно (локальное время в выбранном часовом поясе)
TARGET_START = "2025-09-05 21:30:00"
TARGET_END   = "2025-09-05 22:00:00"
TIMEZONE     = "Europe/Kyiv"

OUTPUT_PATH  = "combined_transformed.csv"
# ===============================================

def _to_dt_utc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s, unit="s", utc=True)

def _linear_stretch(dt_series: pd.Series,
                    src_min: pd.Timestamp,
                    src_max: pd.Timestamp,
                    target_start_utc: pd.Timestamp,
                    target_end_utc: pd.Timestamp) -> pd.Series:
    # Если нет диапазона — всё в начало целевого окна
    if pd.isna(src_min) or pd.isna(src_max) or (src_max == src_min):
        return pd.Series([target_start_utc] * len(dt_series),
                         index=dt_series.index,
                         dtype="datetime64[ns, UTC]")
    src_ns = dt_series.view("int64")
    g0, g1 = src_min.value, src_max.value
    t0, t1 = target_start_utc.value, target_end_utc.value
    scale = (t1 - t0) / (g1 - g0)

    mask = dt_series.notna()
    out = pd.Series(pd.NaT, index=dt_series.index, dtype="datetime64[ns, UTC]")
    mapped_ns = ((src_ns[mask] - g0) * scale + t0).astype("int64")
    out.loc[mask] = pd.to_datetime(mapped_ns, utc=True)
    return out

def main():
    # Локализуем целевое окно в выбранном поясе и переведём в UTC
    start_local = pd.Timestamp(TARGET_START).tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    end_local   = pd.Timestamp(TARGET_END).tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    if pd.isna(start_local) or pd.isna(end_local) or (end_local <= start_local):
        raise ValueError("Проверь TARGET_START/TARGET_END и часовой пояс.")

    start_utc = start_local.tz_convert("UTC")
    end_utc   = end_local.tz_convert("UTC")

    frames = []
    for path in FILES:
        df = pd.read_csv(path)
        if ENTRY_COL not in df.columns or EXIT_COL not in df.columns:
            raise ValueError(f"{path}: нет колонок '{ENTRY_COL}', '{EXIT_COL}'")

        # В UNIX -> UTC
        entry_dt = _to_dt_utc(df[ENTRY_COL])
        exit_dt  = _to_dt_utc(df[EXIT_COL])

        # Диапазон только по текущему файлу
        src_min = entry_dt.min()
        src_max = exit_dt.max()

        # Растяжение в общее окно
        entry_stretched_utc = _linear_stretch(entry_dt, src_min, src_max, start_utc, end_utc)
        exit_stretched_utc  = _linear_stretch(exit_dt,  src_min, src_max, start_utc, end_utc)

        # Перезаписываем человекочитаемыми строками (в локальном поясе) нужного формата
        fmt = "%d.%m.%Y %H:%M:%S"
        df[ENTRY_COL] = entry_stretched_utc.dt.tz_convert(TIMEZONE).dt.strftime(fmt)
        df[EXIT_COL]  = exit_stretched_utc.dt.tz_convert(TIMEZONE).dt.strftime(fmt)

        # Для сортировки позже сохраним служебную колонку (UTC)
        df["_sort_exit_utc"] = exit_stretched_utc

        frames.append(df)

    # Склейка и сортировка по растянутому exit_time (UTC)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(by="_sort_exit_utc").drop(columns=["_sort_exit_utc"]).reset_index(drop=True)

    all_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Сохранено: {OUTPUT_PATH} (строк: {len(all_df)})")

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
