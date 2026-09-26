from __future__ import annotations

from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first
from .addresses import get_saved_address


def get_order_history(address_id: str | None = None) -> dict[str, Any]:
    if not address_id:
        address_id = get_saved_address().get("address_id")
    if not address_id:
        return {"status": "error", "message": "No saved address available to fetch order history"}

    try:
        raw = _call_tool("get_food_orders", {"addressId": address_id})
    except Exception as exc:
        debug("food_mcp", f"get_food_orders FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve order history: {exc}"}

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
