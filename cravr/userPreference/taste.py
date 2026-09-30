"""Learned taste profile (feature 1).

Every pick and every rejection teaches Cravr something, saved in the
user's preference store next to their explicit settings:

- `liked_items` / `liked_restaurants`      -- {name: times picked}
- `disliked_items` / `disliked_restaurants` -- {name: times rejected}
- `rejection_reasons`                       -- {"too_pricey": 2, ...}
- `diet_counts`                             -- {"Veg": 4, "Non-Veg": 0}
- `picked_prices`                           -- last 20 prices picked

From those it derives soft defaults: `dietary_preference` once the last
few picks are all one diet (never overriding one the user set), and
`typical_price_inr` (median pick price). `owner` is the profile key --
the delivery address_id (see store.profile_owner). `taste_signals()` turns the
profile into ranking inputs for signals.practical_score.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from ..debug_utils import debug
from .store import load_preferences, save_preferences

_DIET_INFER_AFTER = 4     # picks, all the same diet, before we infer it
_PRICE_MEMORY = 20


def _norm(name: Any) -> str:
    return str(name or "").strip().lower()


def _bump(counts: dict | None, key: str, by: int = 1) -> dict:
    counts = dict(counts or {})
    if key:
        counts[key] = counts.get(key, 0) + by
    return counts


def _prefs(owner: str | None) -> dict[str, Any]:
    return dict(load_preferences(owner).get("preferences") or {})


def record_pick(owner: str | None, dish: dict[str, Any]) -> dict[str, Any]:
    """A dish the user chose. Never raises -- learning must not break ordering."""
    try:
        prefs = _prefs(owner)
        item, restaurant = _norm(dish.get("item_name")), _norm(dish.get("restaurant_name"))
        update: dict[str, Any] = {
            "liked_items": _bump(prefs.get("liked_items"), item),
            "liked_restaurants": _bump(prefs.get("liked_restaurants"), restaurant),
        }
        # A pick forgives an earlier rejection of the same dish.
        disliked = dict(prefs.get("disliked_items") or {})
        if disliked.pop(item, None) is not None:
            update["disliked_items"] = disliked

        diet = str(dish.get("dietary_type") or "")
        if diet in ("Veg", "Non-Veg"):
            update["diet_counts"] = _bump(prefs.get("diet_counts"), diet)
            recent = (list(prefs.get("recent_diets") or []) + [diet])[-_DIET_INFER_AFTER:]
            update["recent_diets"] = recent
            if (
                len(recent) == _DIET_INFER_AFTER and len(set(recent)) == 1
                and (not prefs.get("dietary_preference") or prefs.get("dietary_preference_inferred"))
            ):
                update["dietary_preference"] = recent[0]
                update["dietary_preference_inferred"] = True

        try:
            price = float(dish.get("price_inr") or 0)
        except (TypeError, ValueError):
            price = 0
        if price > 0:
            prices = (list(prefs.get("picked_prices") or []) + [price])[-_PRICE_MEMORY:]
            update["picked_prices"] = prices
            update["typical_price_inr"] = round(median(prices))
        return save_preferences(owner, **update)
    except Exception as exc:
        debug("taste", f"record_pick failed ({exc})")
        return {}


def record_rejection(
    owner: str | None, dishes: list[dict[str, Any]], reason: str | None = None,
) -> dict[str, Any]:
    """Dishes the user turned down, and why (if known). Never raises."""
    try:
        prefs = _prefs(owner)
        update: dict[str, Any] = {}
        if reason:
            update["rejection_reasons"] = _bump(prefs.get("rejection_reasons"), reason)
        # Only "not for me" marks the dishes themselves -- "too pricey" or "too slow"
        # is about the offer, not the food.
        if reason in (None, "not_for_me"):
            disliked = dict(prefs.get("disliked_items") or {})
            for dish in dishes:
                disliked = _bump(disliked, _norm(dish.get("item_name")))
            update["disliked_items"] = disliked
        return save_preferences(owner, **update) if update else prefs
    except Exception as exc:
        debug("taste", f"record_rejection failed ({exc})")
        return {}


def taste_signals(preference_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Ranking inputs from the saved profile (see signals.practical_score)."""
    prefs = (preference_profile or {}).get("preferences") or {}
    if isinstance(prefs, list):
        prefs = prefs[0] if prefs else {}
    if not isinstance(prefs, dict):
        return {}
    reasons = prefs.get("rejection_reasons") or {}
    return {
        "liked_items": dict(prefs.get("liked_items") or {}),
        "liked_restaurants": dict(prefs.get("liked_restaurants") or {}),
        "disliked_items": dict(prefs.get("disliked_items") or {}),
        "typical_price_inr": prefs.get("typical_price_inr"),
        # Someone who keeps saying "too slow" gets delivery time weighted up.
        "speed_sensitive": int(reasons.get("too_slow") or 0) >= 2,
        "price_sensitive": int(reasons.get("too_pricey") or 0) >= 2,
    }
