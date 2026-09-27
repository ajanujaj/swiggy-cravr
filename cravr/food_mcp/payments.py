from __future__ import annotations

from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first


def _check_success(raw: Any) -> str | None:
    if isinstance(raw, dict) and raw.get("success") is False:
        return _first(raw.get("error") or {}, "message", default="Request failed")
    return None


def get_payment_options(address_id: str) -> dict[str, Any]:
    """Wraps Swiggy's get_payment_options. Cravr only ever offers a user two
    payment families (see cart/checkout.py): Cash/COD, and UPI-by-QR
    (generateUPIQR=true on place_food_order) -- the only UPI path that
    doesn't require picking a specific app/intent, which fits a web chat
    better than deep-linking into a native UPI app. This normalizes down to
    just whether each family is currently offered; `raw` keeps the full
    per-method list for anything else a caller might need later."""
    try:
        raw = _call_tool("get_payment_options", {"addressId": address_id}, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"get_payment_options FAILED: {exc}")
        return {"status": "error", "message": f"Unable to fetch payment options: {exc}"}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"get_payment_options parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected get_payment_options response"}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    cod = data.get("cod") if isinstance(data.get("cod"), dict) else {}
    cod_available = bool(_first(cod, "available", "enabled", default=False)) or bool(cod.get("id"))
    all_methods = data.get("allMethods") if isinstance(data.get("allMethods"), list) else []
    upi_methods = [
        m for m in all_methods
        if isinstance(m, dict)
        and m.get("enabled", True)
        and ("upi" in str(m.get("groupName") or "").lower()
             or "upi" in str(m.get("displayName") or "").lower()
             or m.get("kind") in ("intent", "qr"))
    ]
    result = {
        "status": "success",
        "cod_available": cod_available,
        "upi_available": bool(upi_methods),
        "payment_amount": data.get("paymentAmount"),
        "place_order_tool_name": data.get("placeOrderToolName"),
        "raw": data,
    }
    debug(
        "food_mcp",
        f"get_payment_options parsed: cod_available={result['cod_available']!r} "
        f"upi_available={result['upi_available']!r} payment_amount={result['payment_amount']!r}",
    )
    return result


def place_order(
    address_id: str, payment_method: str, *, generate_upi_qr: bool = False,
    intent_app: str | None = None, note_to_restaurant: str | None = None,
) -> dict[str, Any]:
    """Wraps place_food_order -- THE point of no return: this creates a REAL
    order (Swiggy has no sandbox/dummy-payment mode -- checked their docs).
    `payment_method` is Swiggy's own vocabulary ("Cash" or "UPI"), echoed
    from get_payment_options. Pass generate_upi_qr=True for the scan-to-pay
    path (see get_payment_options's docstring for why Cravr prefers this
    over intentApp deep-linking).

    Returns status="success" (Cash: placed immediately, CONFIRMED) or
    status="pending_payment" (UPI: PENDING_PAYMENT, with paas_id/bridge_url/
    polling fields to hand to check_payment_status) or status="error"."""
    arguments: dict[str, Any] = {"addressId": address_id}
    if payment_method:
        arguments["paymentMethod"] = payment_method
    if generate_upi_qr:
        arguments["generateUPIQR"] = True
    if intent_app:
        arguments["intentApp"] = intent_app
    if note_to_restaurant:
        arguments["noteToRestaurant"] = note_to_restaurant

    try:
        raw = _call_tool("place_food_order", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"place_food_order FAILED: {exc}")
        return {"status": "error", "message": f"Unable to place order: {exc}"}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"place_food_order parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected place_food_order response"}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    swiggy_status = str(data.get("status") or "").upper()
    normalized_status = str(data.get("normalizedStatus") or "").lower()
    is_pending = normalized_status == "pending" or swiggy_status == "PENDING_PAYMENT"

    result: dict[str, Any] = {
        "status": "pending_payment" if is_pending else "success",
        "order_id": data.get("orderId"),
        "swiggy_status": data.get("status"),
        "message": raw.get("message"),
        "raw": data,
    }
    if is_pending:
        result.update({
            "paas_id": data.get("paasId"),
            "transaction_id": data.get("transactionId"),
            "upi_intent_url": data.get("upiIntentUrl"),
            "bridge_url": data.get("bridgeUrl"),
            "is_qr_flow": data.get("isQrFlow"),
            "polling_interval_ms": data.get("pollingIntervalInMs"),
            "max_poll_time_ms": data.get("maxTimeToPollForInMs"),
            "lat": data.get("lat"),
            "lng": data.get("lng"),
            "address_id": data.get("addressId", address_id),
            "cart_id": data.get("cartId"),
        })
    else:
        result.update({
            "items": data.get("items"),
            "restaurant_name": data.get("restaurantName"),
            "restaurant_address": data.get("restaurantAddress"),
            "total_amount": data.get("totalAmount"),
            "estimated_delivery": data.get("estimatedDelivery"),
            "delivery_address": data.get("deliveryAddress"),
        })
    debug("food_mcp", f"place_food_order parsed: status={result['status']!r} order_id={result['order_id']!r}")
    return result


def check_payment_status(
    paas_id: str, order_id: str | None = None, address_id: str | None = None,
    lat: float | None = None, lng: float | None = None, cart_id: str | None = None,
) -> dict[str, Any]:
    """Wraps check_payment_status. For Food, always pass addressId+lat+lng
    (echoed from place_order's response) or reconciliation fails per Swiggy's
    docs. `status` is one of success/paid/failed/refund-initiated/cancelled/
    cart_changed/pending/"" -- `terminal` says whether to stop polling."""
    arguments: dict[str, Any] = {"paasId": paas_id}
    if order_id:
        arguments["orderId"] = order_id
    if address_id:
        arguments["addressId"] = address_id
    if lat is not None:
        arguments["lat"] = lat
    if lng is not None:
        arguments["lng"] = lng
    if cart_id:
        arguments["cartId"] = cart_id

    try:
        raw = _call_tool("check_payment_status", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"check_payment_status FAILED: {exc}")
        return {
            "status": "error", "message": f"Unable to check payment status: {exc}",
            "terminal": True, "is_terminal_success": False, "is_terminal_failure": True,
        }

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"check_payment_status parsed: status='error' message={error!r}")
        return {
            "status": "error", "message": error,
            "terminal": True, "is_terminal_success": False, "is_terminal_failure": True,
        }
    if not isinstance(raw, dict):
        return {
            "status": "error", "message": "Unexpected check_payment_status response",
            "terminal": True, "is_terminal_success": False, "is_terminal_failure": True,
        }

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    result = {
        "status": data.get("status") or "",
        "terminal": bool(data.get("terminal")),
        "is_terminal_success": bool(data.get("isTerminalSuccess")),
        "is_terminal_failure": bool(data.get("isTerminalFailure")),
        "confirmed": data.get("confirmed"),
        "order_status": data.get("orderStatus"),
        "message": raw.get("message"),
    }
    debug("food_mcp", f"check_payment_status parsed: status={result['status']!r} terminal={result['terminal']!r}")
    return result


def confirm_order(
    order_id: str, address_id: str, lat: float | None, lng: float | None,
    cart_id: str | None = None,
) -> dict[str, Any]:
    """Food's confirm_order signature per docs: orderId + addressId + lat +
    lng (NOT paasId -- that's Instamart/Dineout only). Idempotent for the
    same identifiers, so it's safe to call again if unsure an earlier call
    landed. Call only after a terminal-success payment status for UPI, or
    once more at a polling deadline (a late success reconciles server-side)."""
    arguments: dict[str, Any] = {"orderId": order_id, "addressId": address_id, "lat": lat, "lng": lng}
    if cart_id:
        arguments["cartId"] = cart_id

    try:
        raw = _call_tool("confirm_order", arguments, log_request=True)
    except Exception as exc:
        debug("food_mcp", f"confirm_order FAILED: {exc}")
        return {"status": "error", "message": f"Unable to confirm order: {exc}"}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"confirm_order parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected confirm_order response"}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    result_field = str(data.get("result") or "").lower()
    status = result_field if result_field in {"success", "failed", "pending"} else "success"
    result = {
        "status": status,
        "order_id": data.get("orderId", order_id),
        "paas_id": data.get("paasId"),
        "order_status": data.get("orderStatus"),
        "message": raw.get("message"),
    }
    debug("food_mcp", f"confirm_order parsed: status={result['status']!r} order_status={result['order_status']!r}")
    return result
