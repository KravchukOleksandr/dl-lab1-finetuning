# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Core function (concise; English docs & minimal comments) ----------
def compute_hourly_max_concurrent(df_people: pd.DataFrame,
                                  day: pd.Timestamp,
                                  zone_col: str = "zone_of_entrance",
                                  hours=range(0, 24)) -> pd.DataFrame:
    """
    Return a [hour x zone] DataFrame with the maximum number of concurrent people
    in each zone during each hour of a given day.

    Assumes df_people has 'entry_dt'/'exit_dt' (tz-aware datetimes) and a zone column.
    For each hour window [h, h+1), intervals are clipped to the window and a sweep-line
    (entry +1 / exit -1) computes the maximum overlap.
    """
    tmp = df_people[[zone_col, "entry_dt", "exit_dt"]].copy()
    zones = sorted(x for x in tmp[zone_col].dropna().unique() if x != "None")
    out = pd.DataFrame(0, index=list(hours), columns=zones, dtype=int)

    for h in hours:
        start = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                             hour=h, minute=0, second=0, tz=day.tz)
        end = start + pd.Timedelta(hours=1)
        win = tmp[(tmp["entry_dt"] < end) & (tmp["exit_dt"] > start)]
        for z in zones:
            zz = win[win[zone_col] == z]
            if zz.empty:
                continue
            events = []
            for s, e in zip(zz["entry_dt"], zz["exit_dt"]):
                s_clip = max(s, start)
                e_clip = min(e, end)
                if e_clip > s_clip:
                    events.append((s_clip, 1))
                    events.append((e_clip, -1))
            if not events:
                continue
            events.sort(key=lambda x: (x[0], -x[1]))  # entries before exits at same time
            cur = max_cur = 0
            for _, d in events:
                cur += d
                if cur > max_cur:
                    max_cur = cur
            out.loc[h, z] = max_cur
    return out
# -----------------------------------------------------------------------------


def build_dashboard_from_csv(csv_path: str, tz: str = "Europe/Kyiv"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Загрузка и парсинг локального времени (формат DD.MM.YYYY HH:MM:SS)
    df = pd.read_csv(csv_path)
    time_fmt = "%d.%m.%Y %H:%M:%S"
    df["entry_dt"] = pd.to_datetime(df["entry_time"], format=time_fmt).dt.tz_localize(tz)
    df["exit_dt"]  = pd.to_datetime(df["exit_time"],  format=time_fmt).dt.tz_localize(tz)
    df["duration_min"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 60.0

    # Разделим люди/машины
    people_df = df[df["object_type"].isin(["male", "female"])].copy()
    cars_df   = df[df["object_type"] == "car"].copy()

    # Час суток
    people_df["hour"] = people_df["entry_dt"].dt.hour
    cars_df["hour"]   = cars_df["entry_dt"].dt.hour

    # ---------------- 1) Посещаемость людей по входным зонам (почасово) ----------------
    hours = np.arange(0, 24)
    by_hour_zone = (
        people_df[people_df["zone_of_entrance"] != "None"]
        .groupby(["hour", "zone_of_entrance"]).size()
        .unstack(fill_value=0)
        .reindex(hours, fill_value=0)
    )
    plt.figure()
    for col in by_hour_zone.columns:
        plt.plot(hours, by_hour_zone[col].values, marker='o', label=col)
    plt.title("Visitor Count by Entrance Zone (hourly, all days)")
    plt.xlabel("Hour")
    plt.ylabel("Visitors")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---------------- 2) Поток машин по часам ----------------
    cars_by_hour = cars_df.groupby("hour").size().reindex(hours, fill_value=0)
    plt.figure()
    plt.plot(hours, cars_by_hour.values, marker='o')
    plt.title("Vehicle Flow by Hour (all days)")
    plt.xlabel("Hour")
    plt.ylabel("Vehicles")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    # ---------------- 3) Общее число людей по входным зонам (горизонтальные столбцы) ----------------
    total_by_zone = (
        people_df[people_df["zone_of_entrance"] != "None"]["zone_of_entrance"]
        .value_counts()
        .sort_values()
    )
    plt.figure()
    plt.barh(total_by_zone.index, total_by_zone.values)
    plt.title("Total Visitors by Entrance Zone")
    plt.xlabel("Visitors")
    plt.grid(True, axis='x')
    plt.tight_layout()

    # ---------------- 4) Heatmap «Real-Time crowd density» по дням ----------------
    unique_days = sorted(people_df["entry_dt"].dt.normalize().unique())
    for d in unique_days:
        day_ts = pd.Timestamp(d).tz_localize(None).tz_localize(tz)
        occ_max = compute_hourly_max_concurrent(people_df, day_ts, zone_col="zone_of_entrance", hours=range(0, 24))
        plt.figure()
        plt.imshow(occ_max.T, aspect='auto', origin='lower')
        plt.title(f"Real-Time Crowd Density (Max Concurrent) — {day_ts.date()}")
        plt.xlabel("Hour")
        plt.ylabel("Entrance Zone")
        plt.xticks(ticks=np.arange(len(occ_max.index)), labels=occ_max.index)
        plt.yticks(ticks=np.arange(len(occ_max.columns)), labels=occ_max.columns)
        cbar = plt.colorbar()
        cbar.set_label("Max concurrent people")
        plt.tight_layout()

    # ---------------- 5) Среднее время пребывания людей по входным зонам ----------------
    avg_dwell = (
        people_df[people_df["zone_of_entrance"] != "None"]
        .groupby("zone_of_entrance")["duration_min"].mean()
        .sort_index()
    )
    plt.figure()
    plt.bar(avg_dwell.index, avg_dwell.values)
    plt.title("Avg Dwell Time by Entrance Zone (min)")
    plt.ylabel("Minutes")
    plt.grid(True, axis='y')
    plt.tight_layout()

    # ---------------- 6) Время парковки (min / avg / max) ----------------
    car_dur = cars_df["duration_min"]
    if not car_dur.empty:
        car_stats = pd.Series({"min": car_dur.min(), "avg": car_dur.mean(), "max": car_dur.max()})
        plt.figure()
        plt.bar(car_stats.index, car_stats.values)
        plt.title("Vehicle Parking Time (min / avg / max)")
        plt.ylabel("Minutes")
        plt.grid(True, axis='y')
        plt.tight_layout()

    # ---------------- 7) Круговая по полу ----------------
    gender_counts = people_df["object_type"].value_counts().reindex(["female", "male"]).fillna(0)
    plt.figure()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%")
    plt.title("Gender Distribution (People)")
    plt.tight_layout()

    print("Figures created. Call plt.show() to display, or save them as needed.")


# ---------------- Example ----------------
# if __name__ == "__main__":
#     build_dashboard_from_csv("simulated_visits_v2.csv", tz="Europe/Kyiv")
#     import matplotlib.pyplot as plt
#     plt.show()
