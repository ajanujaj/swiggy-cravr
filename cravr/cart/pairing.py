"""Complete the meal (feature 4).

After a dish is in the cart, one tap on "Complete the meal" suggests a
side and a drink from the same restaurant (Swiggy carts are one
restaurant), priced sensibly next to the main, and offers them as a
multi-select that goes through the normal add-to-cart flow.
"""

from __future__ import annotations

import re
from typing import Any

from .. import food_mcp
from ..client import using_real_food_mcp
from ..debug_utils import debug
from ..state import CravrState
from .display import _cart_reply
from .guards import _requires_customization
from .parsing import _existing_cart_items
from .restaurant_context import _current_restaurant_id, _current_restaurant_name

DRINK_WORDS = ("beverage", "drink", "juice", "lassi", "chaas", "buttermilk", "shake", "coke", "pepsi",
               "soda", "cola", "tea", "chai", "coffee", "lemonade", "nimbu", "mojito", "cooler",
               "sprite", "thums", "water", "smoothie", "iced", "ctc", "latte")
SIDE_WORDS = ("side", "sides", "raita", "fries", "naan", "roti", "kulcha", "papad", "salad", "starter",
              "starters", "appetizer", "soup", "garlic bread", "wedges", "nuggets", "dip", "extra",
              "add on", "add-on", "accompaniment", "bread", "rice", "pav", "chutney", "dessert", "sweet",
              "gulab jamun", "brownie", "ice cream")
MAIN_WORDS = ("thali", "combo", "meal", "biryani", "pizza", "burger")

SIDE_MIN_CAP, DRINK_MIN_CAP = 120, 100
PER_KIND = 2


def _text(item: dict[str, Any]) -> str:
    return f"{item.get('item_name') or ''} {item.get('tags') or ''} {item.get('course') or ''}".lower()


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text) for w in words)


def classify(item: dict[str, Any]) -> str | None:
    course = str(item.get("course") or "").lower()
    if course == "beverage":
        return "drink"
    if course in ("side", "dessert"):
        return "side"
    text = _text(item)
    if _has(text, DRINK_WORDS):
        return "drink"
    if _has(text, SIDE_WORDS) and not _has(str(item.get("item_name") or "").lower(), MAIN_WORDS):
        return "side"
    return None


def _price(item: dict[str, Any]) -> float:
    try:
        return float(item.get("price_inr") or 0)
    except (TypeError, ValueError):
        return 0.0


def _rating(item: dict[str, Any]) -> float:
    try:
        return float(str(item.get("rating") or 0).rstrip("★"))
    except (TypeError, ValueError):
        return 0.0


def suggest_pairings(menu: list[dict[str, Any]], main_price: float, exclude_ids: set[str],
                     budget_left: float | None = None) -> list[dict[str, Any]]:
    """Up to two sides and two drinks, cheapest-good-first, within sensible caps."""
    caps = {"side": max(SIDE_MIN_CAP, 0.5 * main_price), "drink": max(DRINK_MIN_CAP, 0.4 * main_price)}
    buckets: dict[str, list[dict[str, Any]]] = {"side": [], "drink": []}
    for item in menu:
        if str(item.get("menu_item_id")) in exclude_ids or _requires_customization(item.get("tags") or ""):
            continue
        kind = classify(item)
        price = _price(item)
        if not kind or price <= 0 or price > caps[kind]:
            continue
        if budget_left is not None and price > budget_left:
            continue
        buckets[kind].append({**item, "pairing": kind})
    picks: list[dict[str, Any]] = []
    for kind in ("side", "drink"):
        ranked = sorted(buckets[kind], key=lambda i: (-_rating(i), _price(i)))
        for item in ranked[:PER_KIND]:
            label = "Goes with your order" if kind == "side" else "To drink"
            picks.append({**item, "reason": f"{label} · ₹{int(_price(item))}"})
    return picks


def _menu_for(restaurant_id: str, restaurant_name: str, address_id: str) -> list[dict[str, Any]]:
    if using_real_food_mcp():
        return food_mcp.get_restaurant_menu_items(restaurant_id, address_id, restaurant_name)
    from ..recommend.RecommendAgent import _call_recommendation_tool

    return _call_recommendation_tool({"restaurant_name": restaurant_name}, address_id=address_id)


def complete_the_meal(state: CravrState, address_id: str) -> dict:
    restaurant_id = _current_restaurant_id(state)
    restaurant_name = _current_restaurant_name(state) or ""
    if not restaurant_id and not restaurant_name:
        return _cart_reply("Add a main dish first, then I can suggest a side and a drink to go with it.")

    cart = food_mcp.get_cart(address_id) if using_real_food_mcp() else {}
    existing = _existing_cart_items(cart) if cart else []
    existing_ids = {str(i["menu_item_id"]) for i in existing}
    picked = [d for d in (state.get("recommendation_state") or {}).get("last_recommendations") or []
              if isinstance(d, dict)]
    main_price = max([_price(d) for d in picked] + [0]) or 250.0

    budget = ((state.get("recommendation_brief") or {}).get("filters") or {}).get("group_budget_inr")
    menu = _menu_for(str(restaurant_id or ""), restaurant_name, address_id)
    pairings = suggest_pairings(menu, main_price, existing_ids | {str(d.get("menu_item_id")) for d in picked},
                                budget_left=(budget - main_price) if budget else None)
    debug("cart_agent", f"complete_meal: menu={len(menu)} pairings={[p.get('item_name') for p in pairings]}")
    if not pairings:
        return _cart_reply(
            f"I couldn't find a side or drink at {restaurant_name or 'this restaurant'} that pairs well "
            "-- your cart is ready whenever you are."
        )

    from ..recommend.RecommendAgent import _offer_dish_batch

    scope = {
        "restaurant_id": restaurant_id, "restaurant_name": restaurant_name,
        "existing_ids": sorted(existing_ids),
    }
    return _offer_dish_batch(
        f"To round off your meal from {restaurant_name or 'the same place'}:", pairings, scope,
    )
