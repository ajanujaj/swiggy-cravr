"""Real Swiggy Food MCP adapter -- maps Cravr's existing agent contracts
onto Swiggy's actual Food MCP tools (get_addresses, search_restaurants,
get_restaurant_menu, get_food_orders). Used instead of the mock server's
`user_location`/`recommendation`/`order_history` tools once
`client.using_real_food_mcp()` is true.

Confirmed against real responses:
- `get_addresses` returns a natural-language block for display (numbered
  addresses with "(ID: ...)" markers and a "Suggested/preselected address
  ID: ..." line), not JSON -- `get_saved_address` parses it with regex.
- `search_restaurants` returns JSON (a `{"restaurants": [...], "dishes": [],
  ...}` object) *followed by* trailing human-readable text (a UI-widget
  disclaimer) appended after it with no separator json.loads recognizes --
  `_parse_leading_json` uses `json.JSONDecoder.raw_decode` to parse just the
  JSON prefix rather than choking on "Extra data". Restaurant objects use
  "id" (not "restaurantId") and "cuisines" (a list, not singular "cuisine").
- `get_restaurant_menu` / `get_food_orders` are still UNVERIFIED -- assumed
  to follow the same JSON-plus-trailing-text convention (now handled by
  `_parse_leading_json` either way); `_normalize_item`'s field-name guesses
  for menu items are still speculative pending a real sample (CRAVR_DEBUG=1
  prints the raw response for each).
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
    """Swiggy's tools return a JSON object/array followed by trailing
    human-readable text -- e.g. search_restaurants' response ends with
    "\n\n\u26a0\ufe0f A rich UI widget may be shown...". Plain
    `json.loads` chokes on that with "Extra data" and silently falls back
    to the raw string, which downstream code then iterates character by
    character (0 results, no error). `raw_decode` parses just the JSON
    prefix and ignores whatever follows. If the text isn't JSON at all
    (get_addresses' plain-language response), this falls back to the raw
    text unchanged, same as before."""
    stripped = text.lstrip()
    try:
        value, _end = _JSON_DECODER.raw_decode(stripped)
        return value
    except json.JSONDecodeError:
        return text


_ADDRESS_LINE_RE = re.compile(
    r"^\d+\.\s*\[(?P<label>[^\]]+)\]\s*[^:]*:\s*(?P<address>.+?)\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)
_SUGGESTED_ID_RE = re.compile(r"Suggested/preselected address ID:\s*([^\s.,]+)")


def get_saved_address() -> dict[str, Any]:
    """-> {"location", "delivery_area", "address_id", "status"}, matching
    ContextAgent.fetch_user_location's contract.

    get_addresses returns a natural-language block for display, e.g.:
        1. [Baba] Anuj Jain: 307, third floor, ... (ID: abc123)
        ...
        Suggested/preselected address ID: abc123. Ask the user to confirm...
    Parsed with regex rather than JSON; uses the suggested ID Swiggy itself
    provides so this stays ambient (no clarification question), matching
    ContextAgent's existing "don't ask the user for it" contract.
    """
    try:
        raw = _unwrap(call_tool_sync("get_addresses", {}))
    except Exception as exc:
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


# Tags (from RecommendAgent.KNOWN_TAGS) that mean "diet food" in Swiggy's
# own EATRIGHT collection sense -- vegan/vegetarian excluded, those map to
# dietary_type/cuisine concerns rather than "healthy eating" as such.
_EATRIGHT_TAGS = {"low-calorie", "high-protein", "low-carb", "gluten-free", "sugar-free-option", "organic"}


def _infer_collection(filters: dict[str, Any]) -> str | None:
    """Optional `collection` scope for search_restaurants -- no BOLT signal
    exists yet (nothing in Cravr captures "make it fast")."""
    tags = {t.strip() for t in str(filters.get("tags") or "").split(",") if t.strip()}
    if tags & _EATRIGHT_TAGS:
        return "EATRIGHT"
    max_price = filters.get("max_price_inr")
    if isinstance(max_price, (int, float)) and max_price <= 99:
        return "STORE_99"
    return None


def search_dishes(filters: dict[str, Any], address_id: str | None, limit_restaurants: int = 5) -> list[dict[str, Any]]:
    """search_restaurants -> get_restaurant_menu -> flatten -> filter
    locally, since Swiggy's tools don't filter by dish attributes.

    `query` is required by the real tool and rejected if empty --
    `_filter_variants` (RecommendAgent.py) progressively drops cuisine/tags
    to relax the search, so by the last attempt(s) neither survives. Falls
    back to dietary_type, then a generic term, rather than failing outright.
    """
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
            menu_raw = _unwrap(call_tool_sync("get_restaurant_menu", {"restaurantId": restaurant_id}))
        except Exception as exc:
            debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
            continue
        debug("food_mcp", f"raw get_restaurant_menu response for {restaurant_id}: {menu_raw!r}")
        items = menu_raw.get("items") if isinstance(menu_raw, dict) else (menu_raw if isinstance(menu_raw, list) else [])
        # Swiggy's search_restaurants shape uses "cuisines" (a list), not
        # "cuisine" -- join it into a readable string for _apply_filters'
        # substring match against the user's requested cuisine.
        restaurant_cuisine = _first(restaurant, "cuisine", "cuisines")
        if isinstance(restaurant_cuisine, list):
            restaurant_cuisine = ", ".join(str(c) for c in restaurant_cuisine)
        for item in items or []:
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


def get_order_history() -> dict[str, Any]:
    """-> {"status", "count", "orders"}, matching preference.py's
    fetch_order_history contract."""
    try:
        raw = _unwrap(call_tool_sync("get_food_orders", {}))
    except Exception as exc:
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
