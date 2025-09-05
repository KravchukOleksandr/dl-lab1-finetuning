# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def generate_visits_db(
    date_str="2025-09-05",
    tz="Europe/Kyiv",
    seed=123,
    # средние длительности людей по зонам (мин)
    zone_mean_minutes={"Gate_A": 4.0, "Main_Hall": 8.0, "Restaurant": 6.0, "Exhibition_Hall": 10.0},
    # доли полов
    p_female=0.40, p_male=0.60,
):
    rng = np.random.default_rng(seed)

    # ---------- Временная сетка ----------
    hours = np.arange(7, 24)  # 7..23 включительно

    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # ---------- Реалистичные почасовые профили людей ----------
    heat_dip = 1 - 0.45 * gaussian(hours, 14.5, 1.6)  # провал в зной

    gate_profile = 0.9*gaussian(hours, 9.5, 0.8) + 1.1*gaussian(hours, 18.7, 1.1) + 0.5*gaussian(hours, 21.0, 0.8)
    hall_profile = 0.6*gaussian(hours, 11.0, 2.2) + 0.9*gaussian(hours, 19.5, 1.8) + 0.3*gaussian(hours, 16.0, 1.6)
    rest_profile = 0.7*gaussian(hours, 13.2, 1.0) + 1.2*gaussian(hours, 20.3, 0.9) + 0.4*gaussian(hours, 10.0, 1.0)
    expo_profile = 0.4*gaussian(hours, 12.0, 2.0) + 1.3*gaussian(hours, 19.0, 1.3) + 0.6*gaussian(hours, 17.0, 1.0)

    def scale_to_counts(profile, scale, base_floor=20, noise_level=3.0):
        """Интенсивность людей/час в диапазоне примерно 20..180."""
        curve = (profile * scale) * heat_dip
        counts = base_floor + curve * 120 + rng.normal(0, noise_level, hours.size)
        # раннее утро слабее
        morning_mask = hours < 9
        counts[morning_mask] *= np.linspace(0.35, 0.7, morning_mask.sum())
        return np.clip(np.rint(counts), 20, 180).astype(int)

    people_hourly_by_zone = {
        "Gate_A":          scale_to_counts(gate_profile, 1.0,  noise_level=3.0),
        "Main_Hall":       scale_to_counts(hall_profile, 0.95, noise_level=4.0),
        "Restaurant":      scale_to_counts(rest_profile, 1.05, noise_level=3.0),
        "Exhibition_Hall": scale_to_counts(expo_profile, 1.1,  noise_level=3.5),
    }

    # ---------- Вспомогательные функции ----------
    def make_timestamp(hour, minute_float):
        """UNIX ts (сек) из локального времени Europe/Kyiv."""
        minute_int = int(minute_float)
        second = int((minute_float - minute_int) * 60)
        ts = pd.Timestamp(f"{date_str} {hour:02d}:{minute_int:02d}:{second:02d}", tz=tz)
        return int(ts.tz_convert("UTC").timestamp())

    def sample_duration_minutes(mean_minutes, min_min=2, max_min=40):
        """Логнормаль вокруг среднего, умеренный разброс."""
        sigma = 0.35
        mu = np.log(mean_minutes) - (sigma**2)/2  # чтобы мат.ожидание≈mean_minutes
        val = rng.lognormal(mean=mu, sigma=sigma)
        return float(np.clip(val, min_min, max_min))

    # ---------- Генерация людей ----------
    records = []
    for zone, hourly_counts in people_hourly_by_zone.items():
        for i, hour in enumerate(hours):
            count = int(hourly_counts[i])
            for _ in range(count):
                # гендерные доли
                obj_type = "female" if rng.random() < p_female else "male"

                # равномерный приход внутри часа (с лёгким смещением для Gate_A)
                frac = rng.random()
                if zone == "Gate_A":
                    if hour < 12:
                        frac = (rng.random()**1.5) * 0.7      # ближе к началу часа утром
                    elif hour >= 18:
                        frac = 0.3 + rng.random()*0.7          # ближе к концу часа вечером

                entry_min = frac * 60.0
                duration_min = sample_duration_minutes(zone_mean_minutes[zone])
                exit_min = entry_min + duration_min

                entry_ts = make_timestamp(int(hour), entry_min)
                exit_hour = int(hour) + int(exit_min // 60)
                exit_minute = exit_min % 60
                exit_hour_clamped = min(exit_hour, 23)
                exit_ts = make_timestamp(exit_hour_clamped, exit_minute if exit_hour <= 23 else 59.0)

                records.append((obj_type, entry_ts, exit_ts, zone))

    # ---------- Генерация машин ----------
    # Менее логичный профиль: шум + вечерние всплески, провал в зной.
    base_cars = rng.integers(5, 25, size=hours.size)
    cars_bump_evening = (10 * gaussian(hours, 20.0, 1.8)).round().astype(int)
    cars_dip_noon = (6 * gaussian(hours, 14.5, 1.2)).round().astype(int)
    car_counts = np.clip(base_cars + cars_bump_evening - cars_dip_noon + rng.integers(-3, 4, size=hours.size), 2, None)

    def sample_car_duration():
        # логнормаль со средней ≈90 мин, границы [5, 300]
        sigma = 0.6
        mu = np.log(90) - (sigma**2)/2
        val = rng.lognormal(mean=mu, sigma=sigma)
        return float(np.clip(val, 5, 300))

    for i, hour in enumerate(hours):
        for _ in range(int(car_counts[i])):
            entry_min = rng.random() * 60.0
            dur = sample_car_duration()
            exit_min = entry_min + dur

            entry_ts = make_timestamp(int(hour), entry_min)
            exit_hour = int(hour) + int(exit_min // 60)
            exit_minute = exit_min % 60
            exit_hour_clamped = min(exit_hour, 23)
            exit_ts = make_timestamp(exit_hour_clamped, exit_minute if exit_hour <= 23 else 59.0)

            records.append(("car", entry_ts, exit_ts, None))  # zone=None для машин

    # ---------- Собираем и сортируем как «живая» база (по времени выхода) ----------
    df = pd.DataFrame(records, columns=["object_type", "entry_time", "exit_time", "zone"])
    df = df.sort_values("exit_time", kind="mergesort").reset_index(drop=True)
    return df

# ------------------------- пример использования -------------------------
if __name__ == "__main__":
    df = generate_visits_db()
    print(df.head(10))
    print("rows:", len(df))
    # Сохранить, если нужно:
    # df.to_csv("simulated_visits.csv", index=False)
