"""get_restaurant_menu -- one restaurant's full menu, plus the parsers/
normalizers `search.py` reuses for its own restaurant-menu lookups."""

from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first

# get_restaurant_menu inlines "[image: <url>]" after almost every line; noise, drop it.
_IMAGE_REF_RE = re.compile(r"\s*\[image:[^\]]*\]")

# get_restaurant_menu's shape: "##"/"###" category headers + dish lines like
# "  - Aloo Paratha ... -- ₹105 | Veg, Bestseller, has addons (ID: 200268099)".
# "###" lines are indented -- the leading "[ \t]*" is required or subcategory
# attribution silently drops.
_MENU_HEADER_RE = re.compile(r"^[ \t]*(#{2,3})\s*(.+?)\s*$", re.MULTILINE)
_MENU_ITEM_RE = re.compile(
    r"^[ \t]*-[ \t]*(?P<name>.+?)(?:[ \t]*—[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?))?"
    r"[ \t]*\|[ \t]*(?P<tags>[^()\n]*?)[ \t]*\(ID:[ \t]*(?P<id>[^)]+)\)[ \t]*$",
    re.MULTILINE,
)
_DIETARY_TAGS = {"veg", "non-veg", "nonveg", "non veg", "egg"}


def _parse_menu_items(
    text: str, restaurant_name: str, restaurant_cuisine: Any, restaurant_id: str | None = None
) -> list[dict[str, Any]]:
    """Parse get_restaurant_menu's text into dish records. Category
    attribution: for each item, take whichever header most recently
    preceded it (## resets subcategory, ### sets it)."""
    headers = [(m.start(), len(m.group(1)), m.group(2)) for m in _MENU_HEADER_RE.finditer(text)]

    def _category_path(pos: int) -> tuple[str, str]:
        category, subcategory = "", ""
        for start, level, title in headers:
            if start >= pos:
                break
            if level == 2:
                category, subcategory = title, ""
            else:
                subcategory = title
        return category, subcategory

    items: list[dict[str, Any]] = []
    for m in _MENU_ITEM_RE.finditer(text):
        raw_tags = [t.strip() for t in m.group("tags").split(",") if t.strip()]
        dietary_tag = next((t for t in raw_tags if t.lower() in _DIETARY_TAGS), "")
        category, subcategory = _category_path(m.start())
        # ";"-joined: _rank_candidates (RecommendAgent.py) splits tags on ";", not ",".
        tags = "; ".join(t for t in (category, subcategory, *raw_tags) if t)

        price_text = m.group("price")
        try:
            price_inr: float | int = float(price_text.replace(",", "")) if price_text else 0
        except ValueError:
            price_inr = 0

        items.append(
            {
                "item_name": m.group("name").strip(),
                "menu_item_id": m.group("id").strip(),
                "restaurant_name": restaurant_name,
                "restaurant_id": restaurant_id,
                "cuisine": restaurant_cuisine or "",
                "price_inr": price_inr,
                "rating": None,
                "tags": tags,
                "dietary_type": "Veg" if dietary_tag.lower() == "veg" else ("Non-Veg" if dietary_tag else ""),
            }
        )
    return items


def _normalize_item(
    item: dict[str, Any], restaurant_name: str, restaurant_cuisine: Any, restaurant_id: str | None = None
) -> dict[str, Any]:
    """JSON-shape counterpart to `_parse_menu_items` -- used on the
    defensive fallback path if a menu response is ever real JSON instead
    of Swiggy's usual plain text."""
    is_veg = _first(item, "isVeg", "veg")
    return {
        "item_name": _first(item, "name", "itemName", default="Item"),
        "menu_item_id": _first(item, "id", "itemId", "menuItemId"),
        "restaurant_name": restaurant_name,
        "restaurant_id": restaurant_id or _first(item, "restaurantId"),
        "cuisine": _first(item, "cuisine", default=restaurant_cuisine) or "",
        "price_inr": _first(item, "price", "finalPrice", "priceInr", default=0),
        "rating": _first(item, "rating"),
        "tags": _first(item, "tags", "category", default=""),
        "dietary_type": "Veg" if is_veg in (True, "true", 1) else ("Non-Veg" if is_veg is not None else ""),
    }


def get_restaurant_menu_items(
    restaurant_id: str, address_id: str | None, restaurant_name: str = ""
) -> list[dict[str, Any]]:
    """get_restaurant_menu for ONE already-known restaurant -- e.g. the one
    already in the cart, when cart_agent wants to suggest more from it.
    Unlike `search.search_dishes`, which loops over several restaurants
    from a fresh search, this fetches exactly one restaurant's full menu.
    Reuses `_parse_menu_items`, the same text parser `search_dishes`
    already relies on for get_restaurant_menu, so category/tags
    attribution (## / ### headers) is identical -- that's what lets a
    caller work out which items are mains vs. sides vs. drinks vs.
    desserts.
    """
    arguments: dict[str, Any] = {"restaurantId": restaurant_id, "addressId": address_id}
    try:
        raw = _call_tool("get_restaurant_menu", arguments, context=f"for {restaurant_id}")
    except Exception as exc:
        debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
        return []

    if isinstance(raw, str):
        raw = _IMAGE_REF_RE.sub("", raw)

    if isinstance(raw, str):
        return _parse_menu_items(raw, restaurant_name, "", restaurant_id)
    if isinstance(raw, dict):
        raw_items = raw.get("items") if isinstance(raw.get("items"), list) else []
        return [
            _normalize_item(item, restaurant_name, "", restaurant_id)
            for item in raw_items if isinstance(item, dict)
        ]
    return []
