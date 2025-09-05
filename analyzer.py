# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Core function (concise English docstring & comments) ----------
def compute_hourly_max_concurrent(df_people: pd.DataFrame,
                                  date_str: str,
                                  tz: str = "Europe/Kyiv",
                                  hours=range(7, 24)) -> pd.DataFrame:
    """
    Return a [hour x zone] DataFrame with the maximum number of concurrent people
    in each zone during each hour.

    The algorithm clips intervals to hour windows and runs a sweep-line over
    entry/exit events inside the window to compute the maximum overlap.
    - df_people: rows with columns ['entry_time','exit_time','zone'] (unix seconds)
    - date_str:  YYYY-MM-DD string for the day to analyze
    - tz:        IANA timezone of the timestamps after conversion from UTC
    - hours:     iterable of hour ints to evaluate (e.g., range(7,24))
    """
    tmp = df_people.copy()
    tmp["entry_dt"] = pd.to_datetime(tmp["entry_time"], unit="s", utc=True).dt.tz_convert(tz)
    tmp["exit_dt"]  = pd.to_datetime(tmp["exit_time"],  unit="s", utc=True).dt.tz_convert(tz)
    zones = sorted(tmp["zone"].dropna().unique())
    out = pd.DataFrame(0, index=list(hours), columns=zones, dtype=int)

    for h in hours:
        start = pd.Timestamp(f"{date_str} {h:02d}:00:00", tz=tz)
        end   = pd.Timestamp(f"{date_str} {h:02d}:59:59.999999", tz=tz)
        window_mask = (tmp["entry_dt"] <= end) & (tmp["exit_dt"] >= start)
        win = tmp[window_mask]
        for z in zones:
            zz = win[win["zone"] == z]
            if zz.empty:
                continue
            # Build events inside the hour window: (+1 at entry, -1 at exit)
            events = []
            for s, e in zip(zz["entry_dt"], zz["exit_dt"]):
                s_clip = max(s, start)
                e_clip = min(e, end)
                if e_clip > s_clip:
                    # tie-break: process entries before exits on equal timestamps
                    events.append((s_clip,  1))
                    events.append((e_clip, -1))
            if not events:
                continue
            events.sort(key=lambda x: (x[0], -x[1]))
            cur = max_conc = 0
            for _, delta in events:
                cur += delta
                if cur > max_conc:
                    max_conc = cur
            out.loc[h, z] = max_conc
    return out
# -------------------------------------------------------------------------

def build_dashboard_from_excel(xlsx_path: str,
                               date_str: str = "2025-09-05",
                               tz: str = "Europe/Kyiv"):
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    # Load data
    df = pd.read_excel(xlsx_path)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], unit="s", utc=True).dt.tz_convert(tz)
    df["exit_dt"]  = pd.to_datetime(df["exit_time"],  unit="s", utc=True).dt.tz_convert(tz)
    df["duration_min"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 60.0

    # Split people / cars
    people_df = df[df["object_type"].isin(["male", "female"])].copy()
    cars_df   = df[df["object_type"] == "car"].copy()

    # Common hour index
    hours = np.arange(7, 24)

    # 1) Visitor Count by Time (per zone)
    people_df["hour"] = people_df["entry_dt"].dt.hour
    visits_by_zone = (
        people_df.groupby(["hour", "zone"]).size()
        .unstack(fill_value=0)
        .reindex(hours, fill_value=0)
    )
    plt.figure()
    for col in visits_by_zone.columns:
        plt.plot(hours, visits_by_zone[col].values, marker='o', label=col)
    plt.title("Visitor Count by Time (7–23)")
    plt.xlabel("Hour")
    plt.ylabel("Visitors")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 2) Vehicle Flow by Hour
    cars_df["hour"] = cars_df["entry_dt"].dt.hour
    cars_by_hour = cars_df.groupby("hour").size().reindex(hours, fill_value=0)
    plt.figure()
    plt.plot(hours, cars_by_hour.values, marker='o')
    plt.title("Vehicle Flow by Hour (7–23)")
    plt.xlabel("Hour")
    plt.ylabel("Vehicles")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    # 3) Unique Visitors per Zone (total)
    total_by_zone = people_df["zone"].value_counts().sort_values()
    plt.figure()
    plt.barh(total_by_zone.index, total_by_zone.values)
    plt.title("Unique Visitors per Zone (total)")
    plt.xlabel("Visitors")
    plt.grid(True, axis='x')
    plt.tight_layout()

    # 4) Real-Time Crowd Density (hourly max concurrent per zone) — heatmap
    occ_max = compute_hourly_max_concurrent(people_df, date_str=date_str, tz=tz, hours=range(7,24))
    plt.figure()
    plt.imshow(occ_max.T, aspect='auto', origin='lower')
    plt.title("Real-Time Crowd Density (Max Concurrent)")
    plt.xlabel("Hour")
    plt.ylabel("Zone")
    plt.xticks(ticks=np.arange(len(occ_max.index)), labels=occ_max.index)
    plt.yticks(ticks=np.arange(len(occ_max.columns)), labels=occ_max.columns)
    cbar = plt.colorbar()
    cbar.set_label("Max concurrent people")
    plt.tight_layout()

    # 5) Avg Dwell Time per Zone (minutes)
    avg_dwell = people_df.groupby("zone")["duration_min"].mean()
    plt.figure()
    plt.bar(avg_dwell.index, avg_dwell.values)
    plt.title("Avg Dwell Time per Zone (min)")
    plt.ylabel("Minutes")
    plt.grid(True, axis='y')
    plt.tight_layout()

    # 6) Vehicle Parking Time — min/avg/max (minutes)
    car_dur = cars_df["duration_min"]
    car_stats = pd.Series({"min": car_dur.min(), "avg": car_dur.mean(), "max": car_dur.max()})
    plt.figure()
    plt.bar(car_stats.index, car_stats.values)
    plt.title("Vehicle Parking Time (min / avg / max)")
    plt.ylabel("Minutes")
    plt.grid(True, axis='y')
    plt.tight_layout()

    # 7) Gender Pie (female vs male count)
    gender_counts = people_df["object_type"].value_counts().reindex(["female", "male"]).fillna(0)
    plt.figure()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%")
    plt.title("Gender Distribution (People)")
    plt.tight_layout()

    print("Done: figures created. Close them or plt.show() to view.")

# ---------------- Example run ----------------
build_dashboard_from_excel("simulated_visits.xlsx", date_str="2025-09-05", tz="Europe/Kyiv")
import matplotlib.pyplot as plt; plt.show()
