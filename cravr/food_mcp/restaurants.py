"""Restaurant facts that decide an order but that `search_menu` doesn't
return: delivery time, offer, distance and whether the place is open.
They come from `search_restaurants`, whose documented response is
`{"data": {"restaurants": [...], "dishes": [...]}}` with `deliveryTimeMinutes`,
`offer`, `distanceKm`, `availabilityStatus` per restaurant.

Lookups are cached for a few minutes per address, since the same handful
of restaurants come up again within one conversation.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first

_CACHE_TTL_SECONDS = 600
_MAX_PARALLEL_LOOKUPS = 5
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _payload(raw: Any, structured: Any) -> dict[str, Any]:
    for source in (structured, raw):
        if not isinstance(source, dict):
            continue
        data = source.get("data") if isinstance(source.get("data"), dict) else source
        if isinstance(data.get("restaurants"), list) or isinstance(data.get("dishes"), list):
            return data
    return {}


def normalize_restaurant(r: dict[str, Any]) -> dict[str, Any]:
    status = str(_first(r, "availabilityStatus", "availability_status", default="OPEN")).upper()
    cuisines = _first(r, "cuisines", "cuisine", default=[])
    if isinstance(cuisines, str):
        cuisines = [c.strip() for c in cuisines.split(",") if c.strip()]
    eta = _to_float(_first(r, "deliveryTimeMinutes", "delivery_time_minutes", "sla"))
    return {
        "restaurant_id": str(_first(r, "id", "restaurantId", "restaurant_id", default="")),
        "restaurant_name": str(_first(r, "name", "restaurantName", "restaurant_name", default="")),
        "cuisines": list(cuisines or []),
        "rating": _to_float(_first(r, "avgRating", "rating")),
        "eta_min": int(eta) if eta else None,
        "eta_range": _first(r, "deliveryTimeRange", default=None),
        "distance_km": _to_float(_first(r, "distanceKm", "distance_km")),
        "offer": _first(r, "offer", "aggregatedDiscountInfo", default=None) or None,
        "cost_for_two": _first(r, "costForTwo", default=None),
        "is_open": status == "OPEN",
    }


def search_restaurants(query: str, address_id: str | None) -> dict[str, Any]:
    """One `search_restaurants` call -> {"restaurants": [...], "dishes": [...]}, normalized."""
    if not address_id or not query:
        return {"restaurants": [], "dishes": []}
    try:
        raw, meta = _call_tool(
            "search_restaurants", {"addressId": address_id, "query": query}, return_meta=True,
        )
    except Exception as exc:
        debug("food_mcp", f"search_restaurants({query!r}) FAILED: {exc}")
        return {"restaurants": [], "dishes": []}

    data = _payload(raw, (meta or {}).get("structured_content"))
    restaurants = [normalize_restaurant(r) for r in data.get("restaurants") or [] if isinstance(r, dict)]
    dishes = []
    for d in data.get("dishes") or []:
        if not isinstance(d, dict):
            continue
        eta = _to_float(_first(d, "deliveryTimeMinutes"))
        dishes.append({
            "item_name": str(_first(d, "name", default="")),
            "menu_item_id": str(_first(d, "id", "menu_item_id", default="")),
            "restaurant_id": str(_first(d, "restaurantId", "restaurant_id", default="")),
            "restaurant_name": str(_first(d, "restaurantName", "restaurant_name", default="")),
            "price_inr": _to_float(_first(d, "price")) or 0,
            "rating": _first(d, "restaurantRating", "rating"),
            "dietary_type": "Veg" if _first(d, "isVeg") else "Non-Veg" if _first(d, "isVeg") is False else "",
            "eta_min": int(eta) if eta else None,
            "tags": _first(d, "category", default="") or "",
            "cuisine": "",
        })
    return {"restaurants": restaurants, "dishes": dishes}


def _lookup_one(restaurant_id: str, restaurant_name: str, address_id: str) -> dict[str, Any] | None:
    result = search_restaurants(restaurant_name, address_id)
    for r in result["restaurants"]:
        if r["restaurant_id"] == restaurant_id:
            return r
    wanted = restaurant_name.strip().lower()
    for r in result["restaurants"]:
        if r["restaurant_name"].strip().lower() == wanted:
            return r
    return None


def restaurant_info(
    restaurants: list[tuple[str, str]], address_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Facts for each (restaurant_id, restaurant_name), keyed by id. Unknown
    restaurants are simply missing -- callers treat that as "no info", not closed."""
    if not address_id:
        return {}
    now = time.time()
    found: dict[str, dict[str, Any]] = {}
    to_fetch: list[tuple[str, str]] = []
    for rid, name in restaurants:
        if not rid or not name or rid in found:
            continue
        cached = _cache.get((address_id, rid))
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            found[rid] = cached[1]
        elif (rid, name) not in to_fetch:
            to_fetch.append((rid, name))

    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_LOOKUPS, len(to_fetch))) as pool:
            results = list(pool.map(lambda pair: _lookup_one(pair[0], pair[1], address_id), to_fetch))
        for (rid, _name), info in zip(to_fetch, results):
            if info:
                _cache[(address_id, rid)] = (now, info)
                found[rid] = info
        debug("food_mcp", f"restaurant info: looked up {len(to_fetch)}, found {len(found)}")
    return found


def remember(address_id: str | None, restaurants: list[dict[str, Any]]) -> None:
    """Cache facts that arrived as a side effect of another search."""
    if not address_id:
        return
    now = time.time()
    for r in restaurants:
        if r.get("restaurant_id"):
            _cache[(address_id, r["restaurant_id"])] = (now, r)
