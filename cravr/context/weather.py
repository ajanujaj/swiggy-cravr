"""Current weather at the delivery address, from Open-Meteo (free, no key).

Swiggy's addresses carry no coordinates, so the city is geocoded from the
address text (last parts first: "..., Pune, Maharashtra, India" -> Pune),
or taken from CRAVR_WEATHER_CITY. Cached 30 minutes per city; any failure
just means "no weather" -- it's a nicety, never a blocker.
Set CRAVR_WEATHER=0 to turn it off.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..debug_utils import debug

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 3.0
_TTL = 1800
_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

# WMO weather codes that mean rain / drizzle / thunderstorm.
_RAIN_CODES = set(range(51, 68)) | set(range(80, 83)) | {95, 96, 99}


def enabled() -> bool:
    return os.getenv("CRAVR_WEATHER", "1").strip().lower() not in ("0", "false", "no", "off")


def _candidate_places(delivery_area: str) -> list[str]:
    parts = [p.strip() for p in (delivery_area or "").replace("(", ",").replace(")", "").split(",")]
    parts = [p for p in parts if p and p.lower() != "india" and not any(ch.isdigit() for ch in p)]
    return list(reversed(parts))[:4]


def classify(temp_c: float | None, weather_code: int | None, precipitation: float | None) -> dict[str, Any]:
    raining = (weather_code in _RAIN_CODES) or (precipitation or 0) > 0.2
    if raining:
        condition, label = "rain", "Rainy"
    elif temp_c is not None and temp_c >= 33:
        condition, label = "hot", "Hot"
    elif temp_c is not None and temp_c <= 15:
        condition, label = "cold", "Chilly"
    else:
        condition, label = "pleasant", "Pleasant"
    return {"condition": condition, "label": label, "temp_c": temp_c}


def _geocode(place: str, client: httpx.Client) -> tuple[float, float] | None:
    response = client.get(_GEOCODE_URL, params={"name": place, "count": 1, "country_code": "IN"})
    results = (response.json() or {}).get("results") or []
    if not results:
        return None
    return float(results[0]["latitude"]), float(results[0]["longitude"])


def current_weather(delivery_area: str | None) -> dict[str, Any] | None:
    if not enabled():
        return None
    places = [os.getenv("CRAVR_WEATHER_CITY")] if os.getenv("CRAVR_WEATHER_CITY") else _candidate_places(delivery_area or "")
    if not places:
        return None
    key = places[0].lower()
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < _TTL:
        return cached[1]

    result = None
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            coords = None
            for place in places:
                coords = _geocode(place, client)
                if coords:
                    break
            if coords:
                response = client.get(_FORECAST_URL, params={
                    "latitude": coords[0], "longitude": coords[1],
                    "current": "temperature_2m,weather_code,precipitation",
                })
                current = (response.json() or {}).get("current") or {}
                result = classify(current.get("temperature_2m"), current.get("weather_code"),
                                  current.get("precipitation"))
    except Exception as exc:
        debug("context_agent", f"weather lookup failed ({type(exc).__name__}: {exc})")
        result = None
    _cache[key] = (time.time(), result)
    debug("context_agent", f"weather for {places[0]!r}: {result}")
    return result
