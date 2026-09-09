"""Real Swiggy Food MCP adapter -- maps Cravr's existing agent contracts
onto Swiggy's actual Food MCP tools (get_addresses, search_restaurants,
get_restaurant_menu, get_food_orders). Used instead of the mock server's
`user_location`/`recommendation`/`order_history` tools once
`client.using_real_food_mcp()` is true.

Swiggy's docs (https://mcp.swiggy.com/builders/docs/reference/food/) list
tool names and one-line descriptions but not exact response schemas, so
the field names in `_normalize_item`/`_first(...)` below are inferred, not
confirmed. Run with CRAVR_DEBUG=1 and check the "raw restaurant/menu/order
response" lines the first time this hits the real server, and adjust the
`_first(...)` key lists here if the actual keys differ.
"""

from __future__ import annotations

import json
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
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
    return result


def get_saved_address() -> dict[str, Any]:
    """-> {"location", "delivery_area", "address_id", "status"}, matching
    ContextAgent.fetch_user_location's contract."""
    try:
        raw = _unwrap(call_tool_sync("get_addresses", {}))
    except Exception as exc:
        return {"status": "error", "message": f"Unable to retrieve saved address: {exc}"}

    debug("food_mcp", f"raw get_addresses response: {raw!r}")
    addresses = raw.get("addresses") if isinstance(raw, dict) else raw
    if not addresses:
        return {"status": "no_results", "message": "No saved addresses found"}

    address = addresses[0]
    return {
        "status": "success",
        "location": _first(address, "city", "area", "label"),
        "delivery_area": _first(address, "addressLine", "display", "label"),
        "address_id": _first(address, "addressId", "id"),
    }


def search_dishes(filters: dict[str, Any], address_id: str | None, limit_restaurants: int = 5) -> list[dict[str, Any]]:
    """search_restaurants -> get_restaurant_menu -> flatten -> filter
    locally, since Swiggy's tools don't filter by dish attributes."""
    query = filters.get("cuisine") or filters.get("tags") or ""
    try:
        raw = _unwrap(call_tool_sync("search_restaurants", {"addressId": address_id, "query": query}))
    except Exception as exc:
        debug("food_mcp", f"search_restaurants FAILED: {exc}")
        return []

    debug("food_mcp", f"raw search_restaurants response: {raw!r}")
    restaurants = raw.get("restaurants") if isinstance(raw, dict) else raw
    if not restaurants:
        return []

    open_restaurants = [
        r for r in restaurants if _first(r, "availabilityStatus", default="OPEN") == "OPEN"
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
        items = menu_raw.get("items") if isinstance(menu_raw, dict) else menu_raw
        for item in items or []:
            dishes.append(_normalize_item(item, restaurant_name, restaurant.get("cuisine") if isinstance(restaurant, dict) else None))

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
    orders = raw.get("orders") if isinstance(raw, dict) else raw
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
