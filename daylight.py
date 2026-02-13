# app/pipeline/daylight_gate.py

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


KYIV_TZ = ZoneInfo("Europe/Kyiv")

# Координаты Киева (если в будущем понадобится — можно вынести в системный конфиг)
KYIV_LAT = 50.4501
KYIV_LON = 30.5234


def _wrap_deg(x: float) -> float:
    return x % 360.0


def _sunrise_sunset_utc(d: date, lat: float, lon: float) -> tuple[datetime | None, datetime | None]:
    """
    Approximate sunrise/sunset (UTC) using NOAA-style formula.
    Returns aware datetimes in UTC or None in edge cases (polar day/night).
    """
    n = d.timetuple().tm_yday
    lng_hour = lon / 15.0

    def _calc(t: float, is_sunrise: bool) -> datetime | None:
        M = (0.9856 * t) - 3.289

        L = M + (1.916 * math.sin(math.radians(M))) + (0.020 * math.sin(math.radians(2 * M))) + 282.634
        L = _wrap_deg(L)

        RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L))))
        RA = _wrap_deg(RA)

        Lq = (math.floor(L / 90.0)) * 90.0
        RAq = (math.floor(RA / 90.0)) * 90.0
        RA = (RA + (Lq - RAq)) / 15.0  # hours

        sin_dec = 0.39782 * math.sin(math.radians(L))
        cos_dec = math.cos(math.asin(sin_dec))

        zenith = 90.833  # official sunrise/sunset incl. refraction
        cos_h = (math.cos(math.radians(zenith)) - (sin_dec * math.sin(math.radians(lat)))) / (
            cos_dec * math.cos(math.radians(lat))
        )

        if cos_h > 1:
            return None  # sun never rises
        if cos_h < -1:
            return None  # sun never sets

        H = (360.0 - math.degrees(math.acos(cos_h))) if is_sunrise else math.degrees(math.acos(cos_h))
        H = H / 15.0  # hours

        T = H + RA - (0.06571 * t) - 6.622
        UT = (T - lng_hour) % 24.0

        hour = int(UT)
        minute = int((UT - hour) * 60.0)
        second = int(round((((UT - hour) * 60.0) - minute) * 60.0))

        dt_utc = datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=timezone.utc) + timedelta(seconds=second)
        return dt_utc

    t_rise = n + ((6.0 - lng_hour) / 24.0)
    t_set = n + ((18.0 - lng_hour) / 24.0)

    return _calc(t_rise, True), _calc(t_set, False)


def is_within_daylight_window(
    now: datetime,
    *,
    buffer_minutes: int = 30,
    tz: ZoneInfo = KYIV_TZ,
    lat: float = KYIV_LAT,
    lon: float = KYIV_LON,
) -> bool:
    """
    True if local time is within [sunrise - buffer, sunset + buffer] for the given date/location.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    now_local = now.astimezone(tz)

    sunrise_utc, sunset_utc = _sunrise_sunset_utc(now_local.date(), lat, lon)

    # Для Киева крайние случаи почти не встречаются; если вдруг встретились — считаем, что можно работать.
    if sunrise_utc is None or sunset_utc is None:
        return True

    sunrise_local = sunrise_utc.astimezone(tz)
    sunset_local = sunset_utc.astimezone(tz)

    buf = timedelta(minutes=buffer_minutes)
    start = sunrise_local - buf
    end = sunset_local + buf
    return start <= now_local <= end


def should_run_motion_check_by_daylight(
    *,
    only_daytime: bool,
    now: datetime,
    buffer_minutes: int = 30,
) -> bool:
    """
    Gate for motion analysis:
    - if only_daytime=False -> always True
    - if only_daytime=True  -> True only during daylight window (Kyiv) +/- buffer
    """
    if not only_daytime:
        return True
    return is_within_daylight_window(now, buffer_minutes=buffer_minutes)