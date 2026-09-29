"""Dishes-first opening for an undecided user.

Instead of asking "what are you in the mood for?" -- the one question an
undecided person can't answer -- Cravr opens with three cards and learns
from the reaction:

1. **Your usual**   -- the dish they order most from their most-ordered
   restaurant, straight from Swiggy's reorder data (skipped if they had it
   in the last day).
2. **Something new** -- a dish they've never ordered, in a category they
   do order, ranked so places they already like come first.
3. **Right now**     -- a dish that fits the time of day (4 PM: vada pav,
   samosa...), ranked by delivery time, offer and rating.

Any card can be missing (no history, search failed); if none can be built
the caller falls back to the usual clarification questions.
"""

from __future__ import annotations

import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from .. import food_mcp
from ..client import using_real_food_mcp
from ..clock import now_ist
from ..debug_utils import debug
from ..intent.meal_clock import current_meal_window
from .signals import history_signals, practical_score

OPENING_KIND = "opening"
SOMETHING_ELSE_VALUE = "something else"
SOMETHING_ELSE_LABEL = "Something else"

TIME_OF_DAY_QUERIES = {
    "breakfast": ["poha", "aloo paratha", "idli", "upma"],
    "lunch": ["thali", "biryani", "rajma chawal", "chole bhature"],
    "evening": ["vada pav", "samosa", "pav bhaji", "momos"],
    "dinner": ["biryani", "paneer butter masala", "butter chicken", "dal makhani"],
    "late_night": ["maggi", "burger", "rolls", "pizza"],
}
WINDOW_PHRASE = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "evening": "an evening snack",
    "dinner": "dinner",
    "late_night": "a late-night bite",
}
# Mock menu has no free-text search; approximate with its own filters.
_MOCK_TIME_FILTERS = {
    "breakfast": {"meal_type": "Breakfast"},
    "lunch": {"course": "main"},
    "evening": {"course": "side,main"},
    "dinner": {"course": "main"},
    "late_night": {"course": "main,dessert"},
}


def _key(dish: dict[str, Any]) -> tuple[str, str]:
    return (
        str(dish.get("restaurant_name") or "").strip().lower(),
        str(dish.get("item_name") or "").strip().lower(),
    )


def _diet_ok(dish: dict[str, Any], dietary_type: str | None) -> bool:
    if not dietary_type:
        return True
    have = str(dish.get("dietary_type") or "")
    return not have or have == dietary_type


# ---------------------------------------------------------------- searching

def _search(query: str | None, mock_filters: dict[str, Any], address_id: str | None,
            dietary_type: str | None) -> list[dict[str, Any]]:
    """Dishes for one opening card. Real Swiggy: `search_restaurants` (dishes
    come with delivery time, and the restaurants' facts get cached), falling
    back to `search_menu`. Mock: the `recommendation` tool."""
    # Imported here: RecommendAgent imports this module's constants.
    from .RecommendAgent import _call_recommendation_tool, _enrich_with_restaurant_info

    if using_real_food_mcp():
        if not query:
            return []
        found = food_mcp.search_restaurants(query, address_id)
        food_mcp.restaurants.remember(address_id, found["restaurants"])
        dishes = [d for d in found["dishes"] if d.get("item_name") and d.get("menu_item_id")]
        if not dishes:
            dishes = food_mcp.search_menu_items(
                query, address_id, veg_only=dietary_type == "Veg",
            ).get("items") or []
        return _enrich_with_restaurant_info(dishes, address_id)

    filters = dict(mock_filters)
    if dietary_type:
        filters["dietary_type"] = dietary_type
    return _call_recommendation_tool(filters, address_id=address_id)


# ---------------------------------------------------------------- the cards

def _usual_pick(history_profile: dict[str, Any], signals: dict[str, Any],
                address_id: str | None) -> dict[str, Any] | None:
    orders = (history_profile or {}).get("orders") or []
    if not orders:
        return None
    recent_items = set(signals.get("recent_items") or [])

    by_restaurant: dict[str, list[dict[str, Any]]] = {}
    for order in orders:  # newest first, so ties go to the most recent restaurant
        name = str(order.get("restaurant_name") or "").strip()
        if name:
            by_restaurant.setdefault(name, []).append(order)

    ranked_restaurants = sorted(by_restaurant, key=lambda r: len(by_restaurant[r]), reverse=True)
    for restaurant in ranked_restaurants:
        restaurant_orders = by_restaurant[restaurant]
        counts: Counter = Counter()
        reorder_by_name: dict[str, dict[str, Any]] = {}
        for order in restaurant_orders:
            names = [i.get("name") for i in (order.get("items") or []) if i.get("name")]
            if not names and order.get("item_name"):
                names = [order["item_name"]]
            counts.update(n.strip() for n in names)
            for r in order.get("reorder_items") or []:
                if r.get("name"):
                    reorder_by_name.setdefault(r["name"].strip().lower(), {**r, "restaurant_id": order.get("restaurant_id")})

        for item_name, times in counts.most_common():
            if item_name.lower() in recent_items:
                continue  # had it within the last day -- not today's usual
            dish = _usual_dish(restaurant, item_name, reorder_by_name.get(item_name.lower()), address_id)
            if dish:
                dish = _with_restaurant_facts(dish, address_id)
                if dish is None:
                    break  # restaurant closed right now -- try their next usual place
                total = len(restaurant_orders)
                dish["badge"] = "Your usual"
                dish["reason"] = (
                    f"You've ordered this {times} time{'s' if times != 1 else ''}"
                    if times > 1 else
                    f"From {restaurant}, where you've ordered {total} time{'s' if total != 1 else ''}"
                )
                return dish
    return None


def _usual_dish(restaurant: str, item_name: str, reorder: dict[str, Any] | None,
                address_id: str | None) -> dict[str, Any] | None:
    """A cart-ready dish for a past order line."""
    if reorder and reorder.get("menu_item_id") and reorder.get("restaurant_id"):
        tags = [t for t, on in (("has variants", reorder.get("has_variants")),
                                ("has addons", reorder.get("has_addons"))) if on]
        return {
            "item_name": reorder.get("name") or item_name,
            "menu_item_id": str(reorder["menu_item_id"]),
            "restaurant_id": str(reorder["restaurant_id"]),
            "restaurant_name": restaurant,
            "price_inr": reorder.get("price_inr") or 0,
            "dietary_type": "Veg" if reorder.get("is_veg") else "Non-Veg",
            "cuisine": reorder.get("category") or "",
            "tags": ", ".join(tags),
        }

    # No reorder data (mock, or an old order): find the item on the menu today.
    if using_real_food_mcp():
        items = food_mcp.search_menu_items(item_name, address_id).get("items") or []
    else:
        from .RecommendAgent import _call_recommendation_tool
        items = _call_recommendation_tool({"restaurant_name": restaurant}, address_id=address_id)
    for item in items:
        if (str(item.get("item_name") or "").strip().lower() == item_name.strip().lower()
                and str(item.get("restaurant_name") or "").strip().lower() == restaurant.strip().lower()):
            return dict(item)
    return None


def _with_restaurant_facts(dish: dict[str, Any], address_id: str | None) -> dict[str, Any] | None:
    """Add delivery time / offer; None when the restaurant is known to be closed."""
    if not using_real_food_mcp():
        return dish
    from .signals import attach_restaurant_info

    info = food_mcp.restaurant_info(
        [(str(dish.get("restaurant_id") or ""), str(dish.get("restaurant_name") or ""))], address_id,
    )
    kept = attach_restaurant_info([dish], info)
    return kept[0] if kept else None


def _new_query(history_profile: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """What "new but still their taste" searches for: their most-ordered
    menu category (real) or cuisine (mock)."""
    categories: Counter = Counter()
    cuisines: Counter = Counter()
    for order in (history_profile or {}).get("orders") or []:
        for r in order.get("reorder_items") or []:
            if r.get("category"):
                categories[str(r["category"]).strip()] += 1
        if order.get("cuisine"):
            cuisines[str(order["cuisine"]).strip()] += 1
    query = categories.most_common(1)[0][0] if categories else None
    mock = {"cuisine": cuisines.most_common(1)[0][0], "course": "main"} if cuisines else {}
    return query, mock


def _best(candidates: list[dict[str, Any]], signals: dict[str, Any], skip: set[tuple[str, str]],
          dietary_type: str | None, never_ordered: bool) -> dict[str, Any] | None:
    ordered = {k for k in (signals.get("item_counts") or {})}
    pool = [
        d for d in candidates
        if _key(d) not in skip
        and _diet_ok(d, dietary_type)
        and not (never_ordered and _key(d)[1] in ordered)
    ]
    if not pool:
        return None
    return dict(max(pool, key=lambda d: practical_score(d, signals)))


def _time_query(window_name: str, now: datetime) -> list[str]:
    queries = TIME_OF_DAY_QUERIES.get(window_name) or ["biryani"]
    start = now.timetuple().tm_yday % len(queries)  # rotate daily so it isn't always the same
    return queries[start:] + queries[:start]


def build_opening_picks(state: dict[str, Any], now: datetime | None = None) -> list[dict[str, Any]]:
    """Up to three cards: usual, something new, right-now. Never raises."""
    now = now or now_ist()
    try:
        return _build(state, now)
    except Exception as exc:  # the opening is a nicety -- never block the conversation on it
        debug("opening", f"could not build opening picks ({type(exc).__name__}: {exc})")
        return []


def _dietary_type(state: dict[str, Any]) -> str | None:
    from .RecommendAgent import DIETARY_TYPE_BY_PREFERENCE, _apply_saved_preferences

    effective = _apply_saved_preferences(state.get("intent_profile") or {}, state.get("preference_profile"))
    for preference in effective.get("dietary_preferences") or []:
        mapped = DIETARY_TYPE_BY_PREFERENCE.get(str(preference).strip().lower())
        if mapped:
            return mapped
    return None


def _build(state: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    address_id = (state.get("context_profile") or {}).get("address_id")
    history_profile = state.get("history_profile") or {}
    signals = history_signals(history_profile, now)
    dietary_type = _dietary_type(state)
    window = current_meal_window(now)

    new_query, new_mock = _new_query(history_profile)
    time_queries = _time_query(window.name, now)

    # The two searches are independent -- run them side by side.
    with ThreadPoolExecutor(max_workers=3) as pool:
        usual_future = pool.submit(_usual_pick, history_profile, signals, address_id)
        new_future = pool.submit(_search, new_query, new_mock, address_id, dietary_type) \
            if (new_query or new_mock) else None
        time_future = pool.submit(_search, time_queries[0], _MOCK_TIME_FILTERS.get(window.name, {}),
                                  address_id, dietary_type)
        usual = usual_future.result()
        new_candidates = new_future.result() if new_future else []
        time_candidates = time_future.result()

    picks: list[dict[str, Any]] = []
    skip: set[tuple[str, str]] = set()
    if usual and _diet_ok(usual, dietary_type):
        picks.append(usual)
        skip.add(_key(usual))

    new = _best(new_candidates, signals, skip, dietary_type, never_ordered=True)
    if new:
        new["badge"] = "Something new"
        restaurant_count = (signals.get("restaurant_counts") or {}).get(_key(new)[0], 0)
        new["reason"] = (
            f"New dish at {new.get('restaurant_name')}, a place you like" if restaurant_count
            else f"Similar to what you usually order{f' ({new_query})' if new_query else ''}"
        )
        picks.append(new)
        skip.add(_key(new))

    now_pick = _best(time_candidates, signals, skip, dietary_type, never_ordered=False)
    if now_pick:
        now_pick["badge"] = f"Good for {WINDOW_PHRASE.get(window.name, 'right now')}"
        now_pick["reason"] = f"It's {now.strftime('%-I:%M %p')}: a good fit for {WINDOW_PHRASE.get(window.name, 'now')}"
        picks.append(now_pick)
        skip.add(_key(now_pick))

    # Short on cards (no history yet)? Top up from the right-now results.
    while len(picks) < 3:
        extra = _best(time_candidates, signals, skip, dietary_type, never_ordered=False)
        if not extra:
            break
        extra["badge"] = "Popular near you"
        extra["reason"] = "Well rated and quick to deliver"
        picks.append(extra)
        skip.add(_key(extra))

    debug("opening", f"opening picks: {[(p.get('badge'), p.get('item_name')) for p in picks]}")
    return picks


def new_opening_token() -> str:
    return uuid.uuid4().hex[:12]
