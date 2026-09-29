from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first

_IMAGE_REF_RE = re.compile(r"\s*\[image:[^\]]*\]")

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


# get_restaurant_menu paginates by CATEGORY (5 per page by default, 8 max
# -- its own inputSchema caps pageSize at 8), and a real restaurant can
# easily have more categories than that: AKU'S - The Burger Co. has 14.
# Only ever fetching page 1 silently drops most of the menu (live
# regression, 2026-09-29: "add more from this restaurant" there only ever
# found ONE survivable category, because 9 of its 14 were never fetched at
# all -- not filtered out, just never seen).
_MENU_PAGE_SIZE = 8
_MENU_MAX_PAGES = 12  # safety cap (96 categories) -- real menus never get close


def _fetch_full_menu(restaurant_id: str, address_id: str | None) -> Any:
    """Loop get_restaurant_menu's pages (using the response's own
    hasMore/page fields, read via _call_tool's return_meta=True) until the
    whole menu has been collected, and concatenate the raw text so the
    existing regex-based _parse_menu_items parser sees one combined menu.
    Each page's "## Category" headers parse independently of page
    boundaries, so naive concatenation is safe."""
    parts: list[str] = []
    page = 1
    while page <= _MENU_MAX_PAGES:
        arguments: dict[str, Any] = {
            "restaurantId": restaurant_id, "addressId": address_id,
            "page": page, "pageSize": _MENU_PAGE_SIZE,
        }
        raw, meta = _call_tool(
            "get_restaurant_menu", arguments, context=f"for {restaurant_id} page {page}",
            return_meta=True,
        )
        if not isinstance(raw, str):
            # Structured/JSON response -- pagination shape unproven here
            # (never observed live); use the single response as-is.
            return raw
        parts.append(raw)
        structured = (meta or {}).get("structured_content")
        if not (isinstance(structured, dict) and structured.get("hasMore")):
            break
        page += 1
    return "\n".join(parts)


def get_restaurant_menu_items(
    restaurant_id: str, address_id: str | None, restaurant_name: str = ""
) -> list[dict[str, Any]]:
    try:
        raw = _fetch_full_menu(restaurant_id, address_id)
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
