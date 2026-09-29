from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first
from .addresses import get_saved_address

# "JB Vadapav (2),Bread Butter (1)" -> [("JB Vadapav", 2), ("Bread Butter", 1)]
_ORDERED_ITEM_RE = re.compile(r"^\s*(?P<name>.+?)\s*(?:\((?P<qty>\d+)\))?\s*$")

# Text fallback, one line per order:
# "1. Order 2373... — Jay Bhavani Vadapav | May 10, 3:45 PM | Delivered | ₹338 [reorderable] — JB Vadapav (2),Ctc (1)"
_ORDER_LINE_RE = re.compile(
    r"^\s*\d+\.\s*Order\s+(?P<id>\S+)\s*[—-]+\s*(?P<restaurant>[^|]+?)\s*\|\s*(?P<time>[^|]+?)\s*\|"
    r"\s*(?P<status>[^|]+?)\s*\|\s*(?P<total>₹?[\d,.]+)[^—\n]*(?:[—-]+\s*(?P<items>.+))?$",
    re.MULTILINE,
)


def _split_ordered_items(text: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in str(text or "").split(","):
        match = _ORDERED_ITEM_RE.match(part)
        if not match or not match.group("name").strip():
            continue
        items.append({"name": match.group("name").strip(), "quantity": int(match.group("qty") or 1)})
    return items


def _reorder_items(order: dict[str, Any]) -> list[dict[str, Any]]:
    """Items from Swiggy's REORDER action -- menu ids, veg flag and category,
    enough to rebuild the cart for a one-tap "your usual"."""
    for action in order.get("actions") or []:
        if not isinstance(action, dict):
            continue
        meta = action.get("reorderMeta")
        if not isinstance(meta, dict):
            continue
        out = []
        for item in meta.get("orderItems") or []:
            if not isinstance(item, dict):
                continue
            category = item.get("categoryDetails") or {}
            try:
                quantity = int(item.get("quantity") or 1)
            except (TypeError, ValueError):
                quantity = 1
            out.append({
                "menu_item_id": str(item.get("itemId") or ""),
                "name": item.get("name") or "",
                "quantity": quantity,
                "is_veg": str(item.get("isVeg")) == "1",
                "category": category.get("category") if isinstance(category, dict) else None,
            })
        return out
    return []


def _normalize_order(order: dict[str, Any]) -> dict[str, Any]:
    reorder = _reorder_items(order)
    items = _split_ordered_items(_first(order, "orderedItems", "ordered_items", default=""))
    if not items and reorder:
        items = [{"name": r["name"], "quantity": r["quantity"]} for r in reorder]
    # Main item = the one ordered in the largest quantity (first wins on ties).
    main_item = max(items, key=lambda i: i["quantity"])["name"] if items else ""
    return {
        "order_id": str(_first(order, "orderId", "order_id", default="")),
        "restaurant_id": str(_first(order, "restaurantId", "restaurant_id", default="")),
        "restaurant_name": str(_first(order, "restaurantName", "restaurant_name", default="")).strip(),
        "cuisine": _first(order, "cuisine", default=""),
        "item_name": _first(order, "itemName", "item_name", default=main_item),
        "items": items,
        "order_total": _first(order, "orderTotal", "order_total", default=""),
        "order_status": _first(order, "orderStatus", "order_status", default=""),
        "order_time": _first(order, "orderedTime", "orderTime", "order_time", "orderDate", "order_date", default=""),
        "meal_type": _first(order, "mealType", "meal_type", default=""),
        "tags": _first(order, "tags", default=""),
        "reorder_items": reorder,
    }


def _parse_order_text(text: str) -> list[dict[str, Any]]:
    orders = []
    for m in _ORDER_LINE_RE.finditer(text or ""):
        orders.append({
            "orderId": m.group("id"),
            "restaurantName": m.group("restaurant"),
            "orderedTime": m.group("time"),
            "orderStatus": m.group("status"),
            "orderTotal": m.group("total"),
            "orderedItems": m.group("items") or "",
        })
    return orders


def _extract_orders(raw: Any, structured: Any) -> list[dict[str, Any]]:
    """Swiggy puts the real order data in `structured_content`; the text
    content is a human-readable summary. Prefer structured, fall back to a
    JSON body, then to parsing the text."""
    for source in (structured, raw):
        if isinstance(source, dict) and isinstance(source.get("orders"), list):
            return [o for o in source["orders"] if isinstance(o, dict)]
        if isinstance(source, list):
            return [o for o in source if isinstance(o, dict)]
    if isinstance(raw, str):
        return _parse_order_text(raw)
    return []


def get_order_history(address_id: str | None = None) -> dict[str, Any]:
    if not address_id:
        address_id = get_saved_address().get("address_id")
    if not address_id:
        return {"status": "error", "message": "No saved address available to fetch order history"}

    try:
        raw, meta = _call_tool("get_food_orders", {"addressId": address_id}, return_meta=True)
    except Exception as exc:
        debug("food_mcp", f"get_food_orders FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve order history: {exc}"}

    orders = _extract_orders(raw, (meta or {}).get("structured_content"))
    if not orders:
        return {"status": "no_results", "message": "No order history found"}

    normalized = [_normalize_order(o) for o in orders]
    debug("food_mcp", f"parsed {len(normalized)} past order(s): {[o['restaurant_name'] for o in normalized]}")
    return {"status": "success", "count": len(normalized), "orders": normalized}
