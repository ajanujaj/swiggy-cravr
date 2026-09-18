"""get_food_cart / update_food_cart / apply_food_coupon / fetch_food_coupons
/ flush_food_cart. Docs describe a consistent {"success", "data",
"message"} JSON envelope for all five, unlike get_addresses/search_menu
(plain text) -- still routed through `_call_tool`/`_unwrap` so a stray
text reply degrades to `_check_success` returning None / an "error" status
instead of crashing, per the search_menu lesson (docs != reality).
"""

from __future__ import annotations

from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first


def _check_success(raw: Any) -> str | None:
    """None when `raw` isn't the documented {"success": false, "error": {...}}
    shape; else the error message to surface."""
    if isinstance(raw, dict) and raw.get("success") is False:
        return _first(raw.get("error") or {}, "message", default="Request failed")
    return None


_DISCLAIMER_MARKERS = (
    "cart widget is displayed",
    "\nnote:",
    "⚠",  # warning-sign emoji prefix on "A rich UI widget may be shown..."
)


def _strip_widget_disclaimer(text: str) -> str:
    """Real cart-tool responses are plain text (confirmed via CRAVR_DEBUG=1
    logs, same lesson as search_menu -- docs describe JSON, reality doesn't):
    a user-facing section (status/items, and for a non-empty cart also a
    pricing breakdown -- Item total/Delivery/Taxes & charges/Coupon/TO PAY,
    itself blank-line-separated from the item list), followed by a blank
    line and agent-only meta-instructions about a "cart widget" our chat UI
    doesn't render ("Cart widget is displayed...", "NOTE: ...", the same "A
    rich UI widget may be shown..." boilerplate search_menu/get_restaurant_menu
    append after their JSON).

    A naive "split on the first blank line" truncates the pricing breakdown
    for a non-empty cart, since that response has THREE blank-line-separated
    paragraphs (items, pricing, disclaimer) not two -- confirmed directly in
    a real get_food_cart/update_food_cart response. So instead we search for
    the earliest occurrence of an actual disclaimer marker and keep
    everything before it; only if no marker is found do we fall back to the
    old first-blank-line split, as a defensive last resort.
    """
    lower = text.lower()
    earliest = None
    for marker in _DISCLAIMER_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest].strip()
    return text.split("\n\n", 1)[0].strip()


def get_cart(address_id: str, restaurant_name: str | None = None) -> dict[str, Any]:
    """get_food_cart -> {"status", "cart_id", "restaurant", "items",
    "item_count", "pricing", "coupon_applied", "coupon_discount", "address_id",
    "message"}. items[]: {"menu_item_id","item_name","quantity","total",
    "final_price","in_stock"} -- populated only on the documented-JSON path
    (e.g. the mock server); the real tool returns plain text with no
    itemized breakdown (see `_strip_widget_disclaimer`), so `message` is
    what actually has the human-readable status there.
    """
    arguments: dict[str, Any] = {"addressId": address_id}
    if restaurant_name:
        arguments["restaurantName"] = restaurant_name

    try:
        raw = _call_tool("get_food_cart", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"get_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve cart: {exc}"}

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        result = {
            "status": "empty" if "cart is empty" in message.lower() else "text",
            "cart_id": None, "restaurant": None, "items": [], "item_count": 0,
            "pricing": {}, "coupon_applied": None, "coupon_discount": 0,
            "address_id": address_id, "message": message,
        }
        debug("food_mcp", f"get_food_cart parsed (text mode): status={result['status']!r} message={message!r}")
        return result

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"get_food_cart parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected get_food_cart response"}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    items = [
        {
            "menu_item_id": _first(it, "menu_item_id", "id", default=""),
            "item_name": _first(it, "name", "item_name", default=""),
            "quantity": _first(it, "quantity", default=1),
            "total": _first(it, "total", default=0),
            "final_price": _first(it, "final_price", "total", default=0),
            "in_stock": it.get("in_stock", True),
        }
        for it in (data.get("items") or []) if isinstance(it, dict)
    ]
    pricing = data.get("pricing") if isinstance(data.get("pricing"), dict) else {}
    offers = data.get("offers") if isinstance(data.get("offers"), dict) else {}
    restaurant = data.get("restaurant") if isinstance(data.get("restaurant"), dict) else None

    result = {
        "status": "success" if items else "empty",
        "cart_id": data.get("cart_id"),
        "restaurant": restaurant,
        "items": items,
        "item_count": data.get("item_count", len(items)),
        "pricing": pricing,
        # coupon_discount==0 means "auto-suggested", not actually applied (per docs).
        "coupon_applied": offers.get("coupon_applied") if (offers.get("coupon_discount") or 0) > 0 else None,
        "coupon_discount": offers.get("coupon_discount", 0),
        "address_id": raw.get("addressId", address_id),
        "message": None,
    }
    debug("food_mcp", f"get_food_cart parsed (json mode): status={result['status']!r} item_count={len(items)}")
    return result


def update_cart(
    restaurant_id: str, address_id: str, cart_items: list[dict[str, Any]], restaurant_name: str | None = None
) -> dict[str, Any]:
    """update_food_cart -- add/change items. Docs: renders no cart view
    itself ("does NOT render any widget or cart UI"), so the documented flow
    is update_food_cart -> get_food_cart; callers here should follow with
    `get_cart` to show the result and confirm the add actually took.

    `cart_items` entries need {"menu_item_id": <id>, "quantity": <int>} --
    NOT bare "id" (an earlier version of this used "id"; a real call with
    it returned 'Cart updated.\nCart is empty.', i.e. accepted but silently
    a no-op, because Swiggy didn't recognise the field. Confirmed via
    Swiggy's own docs, which explicitly tie menu_item_id to this tool:
    '"menu_item_id": Food menu item identifier used by update_food_cart.').

    Items tagged "has variants" still need a real variant/size choice
    Cravr has no UI to collect, and CartAgent refuses those before ever
    calling this. Items tagged "has addons" only are different: CartAgent
    now attempts those with an explicit `"addons": []` entry (see its
    attempt_addon_only) rather than refusing outright -- the request/
    response are always logged here so a real run shows immediately
    whether that guess actually lands.
    """
    arguments: dict[str, Any] = {
        "restaurantId": restaurant_id,
        "addressId": address_id,
        "cartItems": cart_items,
    }
    if restaurant_name:
        arguments["restaurantName"] = restaurant_name

    try:
        raw, meta = _call_tool("update_food_cart", arguments, log_request=True, return_meta=True)
    except Exception as exc:
        debug("food_mcp", f"update_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to update cart: {exc}"}

    # fastmcp's `structured_content` is a SEPARATE channel from the text
    # `raw` above carries -- Swiggy's own machine-readable verdict on the
    # write, not just a human narrative. Live evidence it matters (2026-09-
    # 18, 'Roasted Veggie Pita Pocket'): the text came back 'Cart updated.
    # \nCart is empty.' (which text-mode below would otherwise report as
    # plain success), while `structured_content` had `'successful': False,
    # 'errorCodes': ['INVALID_ADDON'], 'titleMessage': 'Apologies! one or
    # more items in your cart are no longer available'` the whole time.
    # Guarded with isinstance checks throughout -- this field is undocumented
    # (Swiggy's docs don't mention it at all), so nothing about its shape is
    # trusted beyond "a dict, maybe".
    structured = meta.get("structured_content") if isinstance(meta, dict) else None
    explicit_failure = isinstance(structured, dict) and structured.get("successful") is False
    error_codes = structured.get("errorCodes") if explicit_failure else None
    explicit_reason = None
    if explicit_failure:
        explicit_reason = _first(structured, "titleMessage", "statusMessage", default=None)

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        if explicit_failure:
            debug(
                "food_mcp",
                f"update_food_cart parsed (text mode) but structured_content says "
                f"successful=False errorCodes={error_codes!r} reason={explicit_reason!r} -- "
                f"treating as an error despite the 'Cart updated' text",
            )
            return {
                "status": "error",
                "message": explicit_reason or message,
                "error_codes": error_codes,
            }
        debug("food_mcp", f"update_food_cart parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error or explicit_failure:
        final_message = error or explicit_reason or "Request failed"
        debug(
            "food_mcp",
            f"update_food_cart parsed: status='error' message={final_message!r} "
            f"error_codes={error_codes!r}",
        )
        return {"status": "error", "message": final_message, "error_codes": error_codes}
    result = {"status": "success", "message": raw.get("message") if isinstance(raw, dict) else None}
    debug("food_mcp", f"update_food_cart parsed (json mode): {result!r}")
    return result


def apply_coupon(coupon_code: str, address_id: str, cart_id: str | None = None) -> dict[str, Any]:
    """apply_food_coupon -> {"status", "message"}. Caller should re-fetch
    via `get_cart` afterwards to see the updated pricing/coupon_discount."""
    arguments: dict[str, Any] = {"couponCode": coupon_code, "addressId": address_id}
    if cart_id:
        arguments["cartId"] = cart_id

    try:
        raw = _call_tool("apply_food_coupon", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"apply_food_coupon FAILED: {exc}")
        return {"status": "error", "message": f"Unable to apply coupon: {exc}"}

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        debug("food_mcp", f"apply_food_coupon parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"apply_food_coupon parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    result = {"status": "success", "message": raw.get("message") if isinstance(raw, dict) else None}
    debug("food_mcp", f"apply_food_coupon parsed (json mode): {result!r}")
    return result


def fetch_coupons(restaurant_id: str, address_id: str, coupon_code: str | None = None) -> dict[str, Any]:
    """fetch_food_coupons -> {"status", "coupons", "count", "message"}. Docs:
    only COD-applicable coupons are returned. Each coupon: {"id", "title",
    "subtitle", "description", "applicable", "applicability_status", "terms"}
    -- populated only on the documented-JSON path; if the real tool turns
    out to be plain text too (unconfirmed as of this writing, unlike the
    other four cart tools which are all now known to be text), `message`
    carries the raw status and `coupons` stays empty rather than crashing.
    """
    arguments: dict[str, Any] = {"restaurantId": restaurant_id, "addressId": address_id}
    if coupon_code:
        arguments["couponCode"] = coupon_code

    try:
        raw = _call_tool("fetch_food_coupons", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"fetch_food_coupons FAILED: {exc}")
        return {"status": "error", "message": f"Unable to fetch coupons: {exc}", "coupons": []}

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        result = {"status": "text", "message": message, "coupons": []}
        debug("food_mcp", f"fetch_food_coupons parsed (text mode): {result!r}")
        return result

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"fetch_food_coupons parsed: status='error' message={error!r}")
        return {"status": "error", "message": error, "coupons": []}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected fetch_food_coupons response", "coupons": []}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    coupons: list[dict[str, Any]] = []
    for section in data.get("coupon_sections") or []:
        if not isinstance(section, dict):
            continue
        for c in section.get("coupons") or []:
            if not isinstance(c, dict):
                continue
            coupons.append({
                "id": c.get("id"),
                "title": c.get("title"),
                "subtitle": c.get("subtitle"),
                "description": c.get("description"),
                "applicable": c.get("applicable", c.get("applicabilityStatus") == "APPLICABLE"),
                "applicability_status": c.get("applicabilityStatus"),
                "terms": c.get("terms"),
            })

    if not coupons:
        result = {"status": "no_results", "message": data.get("status_message") or "No coupons available", "coupons": []}
        debug("food_mcp", f"fetch_food_coupons parsed (json mode): {result!r}")
        return result
    result = {"status": "success", "coupons": coupons, "count": len(coupons)}
    debug("food_mcp", f"fetch_food_coupons parsed (json mode): {len(coupons)} coupon(s)")
    return result


def flush_cart() -> dict[str, Any]:
    """flush_food_cart -- clears the entire cart. No parameters (auth is
    session-scoped, per docs)."""
    try:
        raw = _call_tool("flush_food_cart", {}, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"flush_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to clear cart: {exc}"}

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        debug("food_mcp", f"flush_food_cart parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"flush_food_cart parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    debug("food_mcp", "flush_food_cart parsed (json mode): success")
    return {"status": "success"}
