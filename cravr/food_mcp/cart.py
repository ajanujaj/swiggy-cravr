from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first


def _check_success(raw: Any) -> str | None:
    if isinstance(raw, dict) and raw.get("success") is False:
        return _first(raw.get("error") or {}, "message", default="Request failed")
    return None


_DISCLAIMER_MARKERS = (
    "cart widget is displayed",
    "\nnote:",
    "⚠",
)


def _strip_widget_disclaimer(text: str) -> str:
    lower = text.lower()
    earliest = None
    for marker in _DISCLAIMER_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest].strip()
    return text.split("\n\n", 1)[0].strip()


def _in_stock(value: Any) -> bool:
    return str(value).strip().lower() not in ("0", "false", "none")


def parse_cart_data(raw: dict[str, Any], address_id: str) -> dict[str, Any]:
    """The cart from a structured get_food_cart / update_food_cart response.
    Both return the whole cart, so an update's response can stand in for a
    fresh get (one Swiggy call instead of two). Each item keeps its add-ons,
    variants and stock flag; `out_of_stock` lists the items Swiggy can't
    deliver as they are (Swiggy says to tell the user before they pay)."""
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    items = [
        {
            "menu_item_id": _first(it, "menu_item_id", "id", default=""),
            "item_name": _first(it, "name", "item_name", default=""),
            "quantity": _first(it, "quantity", default=1),
            "total": _first(it, "total", default=0),
            "final_price": _first(it, "final_price", "total", default=0),
            "in_stock": _in_stock(it.get("in_stock", 1)),
            "addons": [a for a in it.get("addons") or [] if isinstance(a, dict)],
            "variants": [v for v in it.get("variants") or [] if isinstance(v, dict)],
        }
        for it in (data.get("items") or []) if isinstance(it, dict)
    ]
    pricing = data.get("pricing") if isinstance(data.get("pricing"), dict) else {}
    offers = data.get("offers") if isinstance(data.get("offers"), dict) else {}
    restaurant = data.get("restaurant") if isinstance(data.get("restaurant"), dict) else None
    return {
        "status": "success" if items else "empty",
        "cart_id": data.get("cart_id"),
        "restaurant": restaurant,
        "items": items,
        "item_count": data.get("item_count", len(items)),
        "pricing": pricing,
        "coupon_applied": offers.get("coupon_applied") if (offers.get("coupon_discount") or 0) > 0 else None,
        "coupon_discount": offers.get("coupon_discount", 0),
        "address_id": raw.get("addressId", address_id),
        "out_of_stock": [i["item_name"] or str(i["menu_item_id"]) for i in items if not i["in_stock"]],
        "status_message": raw.get("statusMessage"),
        "message": None,
    }


def get_cart(address_id: str, restaurant_name: str | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"addressId": address_id}
    if restaurant_name:
        arguments["restaurantName"] = restaurant_name

    try:
        raw, meta = _call_tool("get_food_cart", arguments, log_request=True, return_meta=True)
    except Exception as exc:
        debug("food_mcp", f"get_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve cart: {exc}"}

    structured = meta.get("structured_content") if isinstance(meta, dict) else None
    if isinstance(structured, dict) and isinstance(structured.get("data"), dict):
        raw = structured

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        out_of_stock = [m.strip() for m in re.findall(r"-\s*(.+?)\s+(?:x\d+\s+)?—[^\n]*OUT OF STOCK", raw)]
        result = {
            "out_of_stock": out_of_stock,
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

    result = parse_cart_data(raw, address_id)
    debug("food_mcp", f"get_food_cart parsed (json mode): status={result['status']!r} item_count={len(result['items'])}")
    return result


def _collect_cart_items(node: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if node.get("menu_item_id") is not None and any(
            isinstance(node.get(k), list) for k in ("valid_addons", "validAddons", "variants", "addons")
        ):
            out.append(node)
        for value in node.values():
            _collect_cart_items(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_cart_items(value, out)


def get_cart_item_details(address_id: str) -> dict[str, dict[str, Any]]:
    try:
        _, meta = _call_tool("get_food_cart", {"addressId": address_id}, log_request=True, return_meta=True)
    except Exception as exc:  # noqa: BLE001 -- best-effort lookup
        debug("food_mcp", f"get_food_cart (item details lookup) FAILED: {exc}")
        return {}
    structured = meta.get("structured_content") if isinstance(meta, dict) else None
    items: list[dict[str, Any]] = []
    _collect_cart_items(structured, items)

    result: dict[str, dict[str, Any]] = {}
    for item in items:
        groups = []
        for g in (item.get("valid_addons") or item.get("validAddons") or []):
            if not isinstance(g, dict):
                continue
            group_id = g.get("group_id") or g.get("groupId")
            choices = []
            for c in g.get("choices") or []:
                if not isinstance(c, dict) or c.get("id") is None:
                    continue
                paise = c.get("price", 0) or 0
                choices.append({
                    "id": str(c["id"]), "name": c.get("name") or "",
                    "price_inr": paise / 100 if isinstance(paise, (int, float)) else 0,
                    "group_id": str(c.get("group_id") or c.get("groupId") or group_id or ""),
                    "raw": c,
                })
            groups.append({
                "group_id": str(group_id) if group_id is not None else None,
                "group_name": g.get("group_name") or g.get("name") or g.get("groupName") or "add-on",
                "min_addons": g.get("minAddons"), "max_addons": g.get("maxAddons"),
                "choices": choices,
            })
        result[str(item["menu_item_id"])] = {
            "variants": [v for v in item.get("variants") or [] if isinstance(v, dict)],
            "addons": [a for a in item.get("addons") or [] if isinstance(a, dict)],
            "valid_addons": groups,
        }
    summary = {
        k: {
            "variants": [(v.get("name"), v.get("variation_id"), v.get("group_id")) for v in d["variants"]],
            "in_cart_addons": [(a.get("name"), a.get("id")) for a in d["addons"]],
            "valid_groups": [
                (g["group_id"], g["group_name"], f"min={g['min_addons']}", f"max={g['max_addons']}",
                 f"{len(g['choices'])} choices")
                for g in d["valid_addons"]
            ],
        }
        for k, d in result.items()
    }
    debug("food_mcp", f"get_food_cart item details -> {summary!r}")
    return result


def get_cart_valid_addons(address_id: str) -> dict[str, list[dict[str, Any]]]:
    return {k: d["valid_addons"] for k, d in get_cart_item_details(address_id).items()}


def update_cart(
    restaurant_id: str, address_id: str, cart_items: list[dict[str, Any]], restaurant_name: str | None = None
) -> dict[str, Any]:
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
        result = {"status": "success", "message": message}
        if isinstance(structured, dict) and isinstance(structured.get("data"), dict):
            result["cart"] = parse_cart_data(structured, address_id)
        return result

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
    for source in (structured, raw):
        if isinstance(source, dict) and isinstance(source.get("data"), dict):
            result["cart"] = parse_cart_data(source, address_id)
            break
    debug("food_mcp", f"update_food_cart parsed (json mode): {result!r}")
    return result


def apply_coupon(coupon_code: str, address_id: str, cart_id: str | None = None) -> dict[str, Any]:
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
