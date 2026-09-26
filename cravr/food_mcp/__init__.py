from __future__ import annotations

from typing import Any

from ..client import call_tool_sync
from ..debug_utils import debug
from ._shared import _unwrap


def _call_tool(
    name: str, arguments: dict[str, Any], *, log_request: bool = False, context: str = "",
    return_meta: bool = False,
) -> Any:
    if log_request:
        debug("food_mcp", f"{name} request: {arguments!r}")

    result = call_tool_sync(name, arguments)

    content = getattr(result, "content", None)
    content_shape = (
        [
            {"type": getattr(item, "type", type(item).__name__), "has_text": getattr(item, "text", None) is not None}
            for item in content
        ]
        if isinstance(content, list)
        else None
    )
    debug(
        "food_mcp",
        f"{name} raw protocol fields -- is_error={getattr(result, 'is_error', None)!r} "
        f"meta={getattr(result, 'meta', None)!r} "
        f"structured_content={getattr(result, 'structured_content', None)!r} "
        f"data={getattr(result, 'data', None)!r} content_shape={content_shape!r}",
    )

    raw = _unwrap(result)
    suffix = f" {context}" if context else ""
    debug("food_mcp", f"raw {name} response{suffix}: {raw!r}")
    if return_meta:
        meta = {
            "is_error": getattr(result, "is_error", None),
            "structured_content": getattr(result, "structured_content", None),
            "data": getattr(result, "data", None),
        }
        return raw, meta
    return raw


from .addresses import get_saved_address
from .menu import _normalize_item, _parse_menu_items, get_restaurant_menu_items
from .search import (
    _infer_collection,
    _parse_detailed_search_menu_items,
    _parse_search_menu_items,
    search_dishes,
    search_menu_items,
)
from .orders import get_order_history
from .cart import (
    _strip_widget_disclaimer,
    apply_coupon,
    fetch_coupons,
    flush_cart,
    get_cart,
    get_cart_item_details,
    get_cart_valid_addons,
    update_cart,
)

__all__ = [
    "call_tool_sync",
    "get_saved_address",
    "get_restaurant_menu_items",
    "search_menu_items",
    "search_dishes",
    "get_order_history",
    "get_cart",
    "get_cart_item_details",
    "get_cart_valid_addons",
    "update_cart",
    "apply_coupon",
    "fetch_coupons",
    "flush_cart",
    "_normalize_item",
    "_parse_menu_items",
    "_infer_collection",
    "_parse_detailed_search_menu_items",
    "_parse_search_menu_items",
    "_strip_widget_disclaimer",
]
