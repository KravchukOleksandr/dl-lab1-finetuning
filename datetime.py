from datetime import datetime
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun

kyiv = LocationInfo(
    name="Kyiv",
    region="Ukraine",
    timezone="Europe/Kyiv",
    latitude=50.4501,
    longitude=30.5234,
)

def is_daylight(now: datetime, buffer_minutes: int = 30) -> bool:
    s = sun(kyiv.observer, date=now.date(), tzinfo=ZoneInfo("Europe/Kyiv"))
    sunrise = s["sunrise"]
    sunset = s["sunset"]

    from datetime import timedelta
    buf = timedelta(minutes=buffer_minutes)

    return (sunrise - buf) <= now <= (sunset + buf)