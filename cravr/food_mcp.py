"""Real Swiggy Food MCP adapter -- maps Cravr's agent contracts onto
Swiggy's Food MCP tools (get_addresses, search_restaurants,
get_restaurant_menu, get_food_orders, all account-scoped via OAuth, no
user_id). Their responses are text, not clean JSON: get_addresses is a
natural-language block; the rest are JSON with trailing human-readable
text appended after it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .client import call_tool_sync
from .debug_utils import debug


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _unwrap(result: Any) -> Any:
    if isinstance(result, (dict, list)):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                return _parse_leading_json(text)
    return result


_JSON_DECODER = json.JSONDecoder()


def _parse_leading_json(text: str) -> Any:
    """Parse a JSON prefix, ignoring trailing text Swiggy appends after it
    (e.g. a UI-widget disclaimer) -- plain json.loads chokes on that with
    "Extra data". Falls back to the raw text if it isn't JSON at all
    (get_addresses' plain-language response)."""
    try:
        value, _end = _JSON_DECODER.raw_decode(text.lstrip())
        return value
    except json.JSONDecodeError:
        return text


_ADDRESS_LINE_RE = re.compile(
    r"^\d+\.\s*\[(?P<label>[^\]]+)\]\s*[^:]*:\s*(?P<address>.+?)\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)
_SUGGESTED_ID_RE = re.compile(r"Suggested/preselected address ID:\s*([^\s.,]+)")

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


def _parse_menu_items(text: str, restaurant_name: str, restaurant_cuisine: Any) -> list[dict[str, Any]]:
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
                "restaurant_name": restaurant_name,
                "cuisine": restaurant_cuisine or "",
                "price_inr": price_inr,
                "rating": None,
                "tags": tags,
                "dietary_type": "Veg" if dietary_tag.lower() == "veg" else ("Non-Veg" if dietary_tag else ""),
            }
        )
    return items


def get_saved_address() -> dict[str, Any]:
    """-> {"location", "delivery_area", "address_id", "status"}, matching
    ContextAgent.fetch_user_location's contract.

    get_addresses returns a natural-language block, e.g.:
        1. [Baba] Anuj Jain: 307, third floor, ... (ID: abc123)
        Suggested/preselected address ID: abc123. Ask the user to confirm...
    Parsed with regex; uses Swiggy's own suggested ID so this stays ambient
    (no clarification question needed).
    """
    try:
        raw = _unwrap(call_tool_sync("get_addresses", {}))
    except Exception as exc:
        debug("food_mcp", f"get_addresses FAILED ({type(exc).__name__}: {exc})")
        return {"status": "error", "message": f"Unable to retrieve saved address: {exc}"}

    debug("food_mcp", f"raw get_addresses response: {raw!r}")
    text = raw if isinstance(raw, str) else json.dumps(raw)

    addresses = {
        m.group("id"): (m.group("label"), m.group("address"))
        for m in _ADDRESS_LINE_RE.finditer(text)
    }
    if not addresses:
        return {"status": "no_results", "message": "No saved addresses found"}

    suggested = _SUGGESTED_ID_RE.search(text)
    address_id = suggested.group(1) if suggested and suggested.group(1) in addresses else next(iter(addresses))
    label, address_line = addresses[address_id]

    return {
        "status": "success",
        "location": label,
        "delivery_area": address_line,
        "address_id": address_id,
    }


# EATRIGHT-flavored tags from RecommendAgent.KNOWN_TAGS (vegan/vegetarian
# excluded -- those are dietary/cuisine, not "healthy eating").
_EATRIGHT_TAGS = {"low-calorie", "high-protein", "low-carb", "gluten-free", "sugar-free-option", "organic"}


def _infer_collection(filters: dict[str, Any]) -> str | None:
    """Optional `collection` scope for search_restaurants. No BOLT mapping --
    nothing in Cravr signals "fast delivery" yet."""
    tags = {t.strip() for t in str(filters.get("tags") or "").split(",") if t.strip()}
    if tags & _EATRIGHT_TAGS:
        return "EATRIGHT"
    max_price = filters.get("max_price_inr")
    if isinstance(max_price, (int, float)) and max_price <= 99:
        return "STORE_99"
    return None


def search_menu_items(
    query: str, address_id: str | None, veg_only: bool = False, offset: int = 0
) -> dict[str, Any]:
    """Search dishes directly by name across restaurants via Swiggy's
    search_menu tool -- more direct than search_restaurants +
    get_restaurant_menu for a specific craving ("sushi", "salad"), and
    supports real pagination (hasMore/nextOffset) instead of fetching
    everything up front.

    Field names below are a best-effort mapping from Swiggy's docs alone (no
    live account to verify against) -- `_first()` tries a few plausible
    aliases per field, and every raw response is logged via `debug()` so a
    wrong guess (a missing dish name, a price off by 100x if it's actually
    in paise) shows up immediately in CRAVR_DEBUG=1 output rather than
    silently returning empty/garbled results.

    Returns {"items": [...normalized dish records...], "has_more": bool,
    "next_offset": int | None}.
    """
    arguments: dict[str, Any] = {"addressId": address_id, "query": query, "offset": offset}
    if veg_only:
        arguments["vegFilter"] = 1

    try:
        raw = _unwrap(call_tool_sync("search_menu", arguments))
    except Exception as exc:
        debug("food_mcp", f"search_menu FAILED: {exc}")
        return {"items": [], "has_more": False, "next_offset": None}

    debug("food_mcp", f"raw search_menu response: {raw!r}")
    if not isinstance(raw, dict):
        return {"items": [], "has_more": False, "next_offset": None}

    items: list[dict[str, Any]] = []
    for entry in raw.get("items") or []:
        if not isinstance(entry, dict):
            continue
        restaurant = entry.get("restaurant") if isinstance(entry.get("restaurant"), dict) else {}
        restaurant_name = _first(entry, "restaurantName") or _first(restaurant, "name", "restaurantName", default="")
        cuisine = _first(entry, "cuisine") or _first(restaurant, "cuisine", "cuisines")
        if isinstance(cuisine, list):
            cuisine = ", ".join(str(c) for c in cuisine)
        name = _first(entry, "name", "itemName", "dishName", default="")
        if not name:
            continue
        items.append({
            "item_name": name,
            "restaurant_name": restaurant_name or "",
            "cuisine": cuisine or "",
            "price_inr": _first(entry, "price", "priceInr", "price_inr", default=0),
            "rating": _first(entry, "rating", "restaurantRating"),
            "tags": _first(entry, "tags", default="") or "",
            "dietary_type": "Veg" if _first(entry, "isVeg", "veg") else "",
        })

    return {
        "items": items,
        "has_more": bool(raw.get("hasMore")),
        "next_offset": raw.get("nextOffset"),
    }


def search_dishes(filters: dict[str, Any], address_id: str | None, limit_restaurants: int = 5) -> list[dict[str, Any]]:
    """search_menu (direct dish-name search) when a specific craving is known --
    e.g. "sushi" from the category/type funnel in intentAgent.py -- else
    search_restaurants -> get_restaurant_menu -> flatten -> filter locally
    (Swiggy's tools don't filter by dish attributes beyond a name/cuisine query).
    """
    craving = filters.get("craving")
    if craving:
        veg_only = filters.get("dietary_type") == "Veg"
        result = search_menu_items(str(craving), address_id, veg_only=veg_only)
        if result["items"]:
            # dietary_type isn't re-checked locally here -- vegFilter above already asked
            # Swiggy to enforce it server-side, and search_menu_items' own veg/isVeg field
            # name is an unverified guess (see its docstring); re-applying a strict local
            # check on a field that might simply be absent could zero out every result.
            local_filters = {k: v for k, v in filters.items() if k != "dietary_type"}
            filtered = _apply_filters(result["items"], local_filters)
            if filtered:
                return filtered
        debug("food_mcp", f"search_menu found nothing usable for {craving!r} -- falling back to search_restaurants")

    query = filters.get("cuisine") or filters.get("tags") or filters.get("dietary_type") or "restaurants"
    arguments: dict[str, Any] = {"addressId": address_id, "query": query}
    collection = _infer_collection(filters)
    if collection:
        arguments["collection"] = collection

    try:
        raw = _unwrap(call_tool_sync("search_restaurants", arguments))
    except Exception as exc:
        debug("food_mcp", f"search_restaurants FAILED: {exc}")
        return []

    debug("food_mcp", f"raw search_restaurants response: {raw!r}")
    restaurants = raw.get("restaurants") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not restaurants:
        return []

    open_restaurants = [
        r for r in restaurants
        if isinstance(r, dict) and _first(r, "availabilityStatus", default="OPEN") == "OPEN"
    ][:limit_restaurants]

    dishes: list[dict[str, Any]] = []
    for restaurant in open_restaurants:
        restaurant_id = _first(restaurant, "restaurantId", "id")
        restaurant_name = _first(restaurant, "name", "restaurantName", default="")
        if not restaurant_id:
            continue
        try:
            # addressId is required here too, same as search_restaurants.
            menu_raw = _unwrap(
                call_tool_sync("get_restaurant_menu", {"restaurantId": restaurant_id, "addressId": address_id})
            )
        except Exception as exc:
            debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
            continue
        if isinstance(menu_raw, str):
            menu_raw = _IMAGE_REF_RE.sub("", menu_raw)
        debug("food_mcp", f"raw get_restaurant_menu response for {restaurant_id}: {menu_raw!r}")
        # "cuisines" is a list, not singular "cuisine" -- join for the filter's substring match.
        restaurant_cuisine = _first(restaurant, "cuisine", "cuisines")
        if isinstance(restaurant_cuisine, list):
            restaurant_cuisine = ", ".join(str(c) for c in restaurant_cuisine)

        if isinstance(menu_raw, str):
            dishes.extend(_parse_menu_items(menu_raw, restaurant_name, restaurant_cuisine))
        else:
            # Defensive fallback if the response is ever real JSON instead of text.
            raw_items = menu_raw.get("items") if isinstance(menu_raw, dict) else (menu_raw if isinstance(menu_raw, list) else [])
            for item in raw_items or []:
                dishes.append(_normalize_item(item, restaurant_name, restaurant_cuisine))

    return _apply_filters(dishes, filters)


def _normalize_item(item: dict[str, Any], restaurant_name: str, restaurant_cuisine: Any) -> dict[str, Any]:
    is_veg = _first(item, "isVeg", "veg")
    return {
        "item_name": _first(item, "name", "itemName", default="Item"),
        "restaurant_name": restaurant_name,
        "cuisine": _first(item, "cuisine", default=restaurant_cuisine) or "",
        "price_inr": _first(item, "price", "finalPrice", "priceInr", default=0),
        "rating": _first(item, "rating"),
        "tags": _first(item, "tags", "category", default=""),
        "dietary_type": "Veg" if is_veg in (True, "true", 1) else ("Non-Veg" if is_veg is not None else ""),
    }


def _apply_filters(dishes: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    dietary_type = filters.get("dietary_type")
    max_price = filters.get("max_price_inr")
    cuisine = (filters.get("cuisine") or "").lower()

    def keep(d: dict[str, Any]) -> bool:
        if dietary_type and str(d.get("dietary_type") or "").lower() != dietary_type.lower():
            return False
        if max_price is not None and isinstance(d.get("price_inr"), (int, float)) and d["price_inr"] > max_price:
            return False
        if cuisine and cuisine not in str(d.get("cuisine") or "").lower():
            return False
        return True

    return [d for d in dishes if keep(d)]


def get_order_history(address_id: str | None = None) -> dict[str, Any]:
    """-> {"status", "count", "orders"}, matching preference.py's
    fetch_order_history contract.

    addressId is required. If the caller doesn't have one yet (usually
    true -- this runs before context_agent resolves it), resolve it here,
    deliberately uncached (see _resolve_address_id in RecommendAgent.py).
    """
    if not address_id:
        address_id = get_saved_address().get("address_id")
    if not address_id:
        return {"status": "error", "message": "No saved address available to fetch order history"}

    try:
        raw = _unwrap(call_tool_sync("get_food_orders", {"addressId": address_id}))
    except Exception as exc:
        debug("food_mcp", f"get_food_orders FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve order history: {exc}"}

    debug("food_mcp", f"raw get_food_orders response: {raw!r}")
    orders = raw.get("orders") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not orders:
        return {"status": "no_results", "message": "No order history found"}

    normalized = [
        {
            "restaurant_name": _first(o, "restaurantName", "restaurant_name", default=""),
            "cuisine": _first(o, "cuisine", default=""),
            "item_name": _first(o, "itemName", "item_name", default=""),
            "meal_type": _first(o, "mealType", "meal_type", default=""),
            "order_date": _first(o, "orderDate", "order_date", default=""),
            "order_time": _first(o, "orderTime", "order_time", default=""),
            "tags": _first(o, "tags", default=""),
        }
        for o in orders
    ]
    return {"status": "success", "count": len(normalized), "orders": normalized}
