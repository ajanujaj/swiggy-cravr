"""A reason on every card (feature 3).

Undecided people need a reason to tap. Each dish gets one short line
built from what we actually know -- their history, what they asked for,
delivery time, offer, rating, budget, context -- best reason first, at
most two joined by " · ". Cards that already carry a reason (the opening
cards) keep theirs.
"""

from __future__ import annotations

from typing import Any


def _rating(dish: dict[str, Any]) -> float:
    try:
        return float(str(dish.get("rating") or 0).rstrip("★"))
    except (TypeError, ValueError):
        return 0.0


def _price(dish: dict[str, Any]) -> float:
    try:
        return float(dish.get("price_inr") or 0)
    except (TypeError, ValueError):
        return 0.0


def card_reason(dish: dict[str, Any], signals: dict[str, Any] | None = None,
                filters: dict[str, Any] | None = None) -> str:
    signals = signals or {}
    filters = filters or {}
    item = str(dish.get("item_name") or "").strip().lower()
    restaurant_name = str(dish.get("restaurant_name") or "").strip()
    restaurant = restaurant_name.lower()
    reasons: list[str] = []

    picked = (signals.get("liked_items") or {}).get(item, 0)
    ordered = (signals.get("item_counts") or {}).get(item, 0)
    visits = max((signals.get("restaurant_counts") or {}).get(restaurant, 0),
                 (signals.get("liked_restaurants") or {}).get(restaurant, 0))
    if picked or ordered:
        times = max(picked, ordered)
        reasons.append("You've had this before" if times == 1 else f"You've had this {times} times")
    elif visits:
        reasons.append(f"From {restaurant_name}, a regular of yours")

    if signals.get("context_label"):
        reasons.append(str(signals["context_label"]))

    craving = str(filters.get("craving") or "").strip()
    if craving and craving.lower() in item and not reasons:
        reasons.append(f"Matches “{craving}”")

    eta = dish.get("eta_min")
    if isinstance(eta, (int, float)) and 0 < eta <= 25:
        reasons.append(f"Arrives in ~{int(eta)} min")
    if dish.get("offer"):
        reasons.append(str(dish["offer"]))

    rating = _rating(dish)
    if rating >= 4.5:
        reasons.append(f"Rated {rating:g}")

    cap = filters.get("max_price_inr")
    price = _price(dish)
    if cap and price and price <= cap:
        reasons.append(f"Within your ₹{int(cap)} budget")
    elif signals.get("typical_price_inr") and price and price <= 0.8 * float(signals["typical_price_inr"]):
        reasons.append("Cheaper than you usually spend")

    if not reasons and rating >= 4.0:
        reasons.append(f"Well rated ({rating:g})")
    return " · ".join(reasons[:2])


def with_reasons(dishes: list[dict[str, Any]], signals: dict[str, Any] | None,
                 filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    for dish in dishes:
        if dish.get("reason"):
            out.append(dish)
            continue
        reason = card_reason(dish, signals, filters)
        out.append({**dish, "reason": reason} if reason else dish)
    return out
