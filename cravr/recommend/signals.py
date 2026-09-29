"""Ranking signals beyond "does it match what they asked": what the user
usually orders, what they ate very recently, and the practical facts that
decide an order -- delivery time, an offer, whether the place is open.

`practical_score` folds them into one number; `_rank_candidates` uses it
right after relevance (craving match, course, tags, cuisine), so history
and speed decide between dishes that fit the request equally well.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ..clock import IST, now_ist

# An item ordered within this window is "just had it" -- demoted, not hidden.
RECENT_WINDOW = timedelta(hours=20)

WEIGHTS = {
    "usual_restaurant": 1.2,   # history is the strongest signal for an undecided user
    "usual_item": 0.6,
    "usual_cuisine": 0.3,
    "eta": 0.6,
    "offer": 0.4,
    "rating": 0.8,
    "recent_item": -3.0,       # had exactly this in the last day: outweighs every other boost
    "recent_restaurant": -0.4,
}

_MONTH_DAY_RE = re.compile(r"^\s*(?P<md>[A-Za-z]+\s+\d{1,2}),?\s*(?P<clock>\d{1,2}:\d{2}\s*[AaPp][Mm])?")


def parse_order_time(order: dict[str, Any], now: datetime | None = None) -> datetime | None:
    """Swiggy: "May 10, 3:45 PM" (no year). Mock: order_date "2026-01-12" + order_time "20:14"."""
    now = now or now_ist()
    date_text = str(order.get("order_date") or "").strip()
    time_text = str(order.get("order_time") or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_text):
        try:
            parsed = datetime.strptime(f"{date_text} {time_text or '00:00'}", "%Y-%m-%d %H:%M")
            return parsed.replace(tzinfo=IST)
        except ValueError:
            return None

    match = _MONTH_DAY_RE.match(time_text or date_text)
    if not match:
        return None
    clock = (match.group("clock") or "12:00 PM").upper().replace(" ", "")
    for fmt in ("%B %d %Y %I:%M%p", "%b %d %Y %I:%M%p"):
        try:
            parsed = datetime.strptime(f"{match.group('md')} {now.year} {clock}", fmt).replace(tzinfo=IST)
        except ValueError:
            continue
        # No year given: a date "in the future" was last year.
        if parsed > now + timedelta(days=1):
            parsed = parsed.replace(year=now.year - 1)
        return parsed
    return None


def _order_item_names(order: dict[str, Any]) -> set[str]:
    names = {str(i.get("name")).strip().lower() for i in (order.get("items") or []) if i.get("name")}
    if not names and order.get("item_name"):
        names = {str(order["item_name"]).strip().lower()}
    return names


def history_signals(history_profile: dict[str, Any] | None, now: datetime | None = None) -> dict[str, Any]:
    """Counts and recency from past orders, as plain lists/dicts (it lives in graph state)."""
    now = now or now_ist()
    orders = (history_profile or {}).get("orders") or []
    restaurants: Counter = Counter()
    items: Counter = Counter()
    cuisines: Counter = Counter()
    recent_items: set[str] = set()
    recent_restaurants: set[str] = set()
    for order in orders:
        restaurant = str(order.get("restaurant_name") or "").strip().lower()
        names = _order_item_names(order)
        if restaurant:
            restaurants[restaurant] += 1
        items.update(names)
        if order.get("cuisine"):
            cuisines[str(order["cuisine"]).strip().lower()] += 1
        when = parse_order_time(order, now)
        if when and now - when <= RECENT_WINDOW:
            recent_items.update(names)
            if restaurant:
                recent_restaurants.add(restaurant)
    return {
        "restaurant_counts": dict(restaurants),
        "item_counts": dict(items),
        "cuisine_counts": dict(cuisines),
        "recent_items": sorted(recent_items),
        "recent_restaurants": sorted(recent_restaurants),
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _rating(dish: dict[str, Any]) -> float:
    try:
        return float(str(dish.get("rating") or 0).rstrip("★"))
    except (TypeError, ValueError):
        return 0.0


def practical_score(dish: dict[str, Any], signals: dict[str, Any] | None) -> float:
    """Higher is better. Every part is optional: a dish with no ETA or
    offer data (mock menu, lookup failed) just gets no credit for them."""
    signals = signals or {}
    restaurant = str(dish.get("restaurant_name") or "").strip().lower()
    item = str(dish.get("item_name") or "").strip().lower()
    cuisine = str(dish.get("cuisine") or "").strip().lower()

    restaurant_count = (signals.get("restaurant_counts") or {}).get(restaurant, 0)
    item_count = (signals.get("item_counts") or {}).get(item, 0)
    cuisine_hit = any(c and c in cuisine for c in (signals.get("cuisine_counts") or {})) if cuisine else False

    score = 0.0
    score += WEIGHTS["usual_restaurant"] * _clamp(restaurant_count / 3)
    score += WEIGHTS["usual_item"] * _clamp(item_count / 3)
    score += WEIGHTS["usual_cuisine"] * (1 if cuisine_hit else 0)

    eta = dish.get("eta_min")
    if isinstance(eta, (int, float)) and eta > 0:
        score += WEIGHTS["eta"] * _clamp((60 - eta) / 40)  # <=20 min full credit, 60+ none
    if dish.get("offer"):
        score += WEIGHTS["offer"]
    rating = _rating(dish)
    if rating:
        score += WEIGHTS["rating"] * _clamp((rating - 3.5) / 1.3)

    if item and item in set(signals.get("recent_items") or []):
        score += WEIGHTS["recent_item"]
    elif restaurant and restaurant in set(signals.get("recent_restaurants") or []):
        score += WEIGHTS["recent_restaurant"]
    return round(score, 4)


def attach_restaurant_info(dishes: list[dict[str, Any]], info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy delivery time / offer / open status onto each dish and drop dishes
    from restaurants known to be closed. No info = kept as is."""
    kept = []
    for dish in dishes:
        facts = info.get(str(dish.get("restaurant_id") or ""))
        if facts:
            if facts.get("is_open") is False:
                continue
            dish = {
                **dish,
                "eta_min": dish.get("eta_min") or facts.get("eta_min"),
                "offer": dish.get("offer") or facts.get("offer"),
                "distance_km": facts.get("distance_km"),
            }
        kept.append(dish)
    return kept
