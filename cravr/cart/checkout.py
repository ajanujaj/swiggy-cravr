from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from .. import food_mcp
from ..debug_utils import debug, test_mode_enabled
from ..options import resolve_option_choice
from .display import _cart_reply, _item_lines
from .routing import _UPI_CONFIRM_OPTIONS

_PAY_COD = "cod"
_PAY_UPI = "upi"
_PAY_CLEAR_CART = "__checkout_clear_cart__"
_PAY_CANCEL = "__cancel_checkout__"

_DEFAULT_POLL_INTERVAL_MS = 4000
_MAX_TOTAL_POLL_MS = 3 * 60 * 1000

_FAKE_ORDER_ID = "TEST-ORDER-0001"


def _order_summary(cart: dict[str, Any], address: dict[str, Any], item_details: dict[str, Any] | None = None) -> str:
    items = cart.get("items") or []
    if items:
        sections = ["**Your order:**\n\n" + "\n".join(_item_lines(items, item_details))]
    else:
        sections = [cart.get("message") or ""]
    pricing = cart.get("pricing") or {}
    to_pay = pricing.get("to_pay")
    if to_pay is not None:
        sections.append(f"**Total to pay:** ₹{to_pay}")
    if address.get("status") == "success":
        sections.append(f"**Delivering to:** {address.get('location')} — {address.get('delivery_area')}")
    return "\n\n".join(sections)


def _payment_method_options(payment_options: dict[str, Any]) -> list[dict[str, str]]:
    unknown = payment_options.get("status") != "success"
    options = []
    if unknown or payment_options.get("cod_available"):
        options.append({"value": _PAY_COD, "label": "Cash on Delivery"})
    if unknown or payment_options.get("upi_available"):
        options.append({"value": _PAY_UPI, "label": "UPI (scan to pay)"})
    options.append({"value": _PAY_CLEAR_CART, "label": "Clear my cart"})
    options.append({"value": _PAY_CANCEL, "label": "Cancel — don't place the order"})
    return options


def run_checkout(address_id: str) -> dict:
    cart = food_mcp.get_cart(address_id)
    debug(
        "cart_agent",
        f"checkout: get_cart -> status={cart.get('status')!r} item_count={len(cart.get('items') or [])}",
    )
    if cart.get("status") == "error":
        return _cart_reply(f"Couldn't load your cart: {cart.get('message', 'unknown error')}")
    if cart.get("status") == "empty" or not cart.get("items"):
        return _cart_reply("Your cart's empty -- add something first, then come back to check out.")

    address = food_mcp.get_saved_address()
    item_details = food_mcp.get_cart_item_details(address_id)
    payment_options = food_mcp.get_payment_options(address_id)
    if payment_options.get("status") != "success":
        debug("cart_agent", f"checkout: get_payment_options FAILED: {payment_options.get('message')!r}")
    options = _payment_method_options(payment_options)
    if not any(o["value"] in (_PAY_COD, _PAY_UPI) for o in options):
        return _cart_reply(
            "Checkout isn't available for this cart right now (Swiggy isn't offering Cash or UPI for it) "
            "-- try again in a bit, or finish the order in the Swiggy app."
        )

    summary = _order_summary(cart, address, item_details)
    raw_answer = interrupt({
        "question": f"{summary}\n\nHow would you like to pay?",
        "options": options,
        "agent": "cart_agent",
        "field": "payment_method",
    })
    debug("cart_agent", f"checkout: payment choice raw_answer={raw_answer!r}")
    choice = resolve_option_choice(str(raw_answer), options)

    if choice == _PAY_CLEAR_CART:
        return _clear_cart_at_checkout()

    if choice not in (_PAY_COD, _PAY_UPI):
        return _cart_reply("Okay, I didn't place the order -- your cart's untouched.")

    if test_mode_enabled():
        return _run_test_mode_order(choice, cart)

    return _run_real_order(choice, address_id)


def _clear_cart_at_checkout() -> dict:
    result = food_mcp.flush_cart()
    debug("cart_agent", f"checkout: 'Clear my cart' button -> flush_cart {result!r}")
    if result.get("status") != "success":
        return _cart_reply(f"Couldn't clear the cart: {result.get('message', 'unknown error')}")
    return _cart_reply(result.get("message") or "Your cart is now empty.", items=[], pricing={})


def _run_test_mode_order(choice: str, cart: dict[str, Any]) -> dict:
    debug(
        "cart_agent",
        f"checkout: CRAVR_TEST_MODE is on -- faking a {choice!r} order, NOT calling "
        "place_food_order/confirm_order/check_payment_status",
    )
    pricing = cart.get("pricing") or {}
    method_label = "Cash on Delivery" if choice == _PAY_COD else "UPI"
    content = (
        "🧪 TEST MODE — no real order was placed, no money moved, nothing was sent to Swiggy.\n\n"
        f"Order summary (fake): {method_label}, total ₹{pricing.get('to_pay', '?')}, "
        f"order id {_FAKE_ORDER_ID}, status CONFIRMED.\n"
        "Unset CRAVR_TEST_MODE to place real orders."
    )
    return _cart_reply(content, restaurant=None)


def _run_real_order(choice: str, address_id: str) -> dict:
    payment_method = "Cash" if choice == _PAY_COD else "UPI"
    order = food_mcp.place_order(address_id, payment_method, generate_upi_qr=(choice == _PAY_UPI))
    debug(
        "cart_agent",
        f"checkout: place_food_order -> status={order.get('status')!r} order_id={order.get('order_id')!r}",
    )
    if order.get("status") == "error":
        return _cart_reply(f"Couldn't place the order: {order.get('message', 'unknown error')}")

    if order.get("status") == "success":
        return _order_placed_reply(order, "Cash on Delivery")

    bridge_url = order.get("bridge_url")
    content = f"Order created (ID: {order.get('order_id')}) — complete the payment to confirm it."
    if bridge_url:
        content += f"\n\n[Tap here to pay]({bridge_url})"
    content += "\n\nOnce you've paid, tap \"I've paid\" and I'll confirm your order."
    return _cart_reply(content, restaurant=None, pending_upi_order=order, options=_UPI_CONFIRM_OPTIONS)


def resume_upi_payment(cart_state: dict[str, Any], address_id: str) -> dict:
    order = cart_state.get("pending_upi_order")
    if not isinstance(order, dict) or not order.get("paas_id"):
        return _cart_reply("I don't have a pending UPI payment to check -- try checking out again.")
    debug("cart_agent", f"checkout: resuming UPI confirmation for order_id={order.get('order_id')!r} -- polling")
    final_status = _poll_payment_status(order, address_id)
    return _finish_upi_order(order, final_status, address_id)


def cancel_upi_payment(cart_state: dict[str, Any]) -> dict:
    order = cart_state.get("pending_upi_order") or {}
    order_id = order.get("order_id", "")
    return _cart_reply(
        f"Okay, I won't confirm order {order_id} -- if you already paid, check the Swiggy app; "
        "otherwise your cart's still here."
    )


def _poll_payment_status(order: dict[str, Any], address_id: str) -> dict[str, Any]:
    interval_ms = order.get("polling_interval_ms") or _DEFAULT_POLL_INTERVAL_MS
    max_ms = min(order.get("max_poll_time_ms") or _MAX_TOTAL_POLL_MS, _MAX_TOTAL_POLL_MS)
    deadline = time.monotonic() + max_ms / 1000
    status: dict[str, Any] = {}
    while True:
        status = food_mcp.check_payment_status(
            order.get("paas_id"), order_id=order.get("order_id"), address_id=address_id,
            lat=order.get("lat"), lng=order.get("lng"), cart_id=order.get("cart_id"),
        )
        debug("cart_agent", f"checkout: check_payment_status -> {status!r}")
        if status.get("terminal") or time.monotonic() >= deadline:
            return status
        time.sleep(interval_ms / 1000)


def _finish_upi_order(order: dict[str, Any], status: dict[str, Any], address_id: str) -> dict:
    state_status = status.get("status")

    if status.get("is_terminal_success") or state_status in ("success", "paid"):
        confirm = food_mcp.confirm_order(
            order.get("order_id"), address_id, order.get("lat"), order.get("lng"), cart_id=order.get("cart_id"),
        )
        debug("cart_agent", f"checkout: confirm_order -> {confirm!r}")
        return _order_placed_reply(order, "UPI", confirm=confirm)

    if state_status == "cart_changed":
        return _cart_reply(
            "Your cart changed before the payment went through -- take a look and try checking out again."
        )
    if state_status == "cancelled":
        return _cart_reply("Payment was cancelled -- your cart's still here whenever you're ready.")
    if state_status == "refund-initiated":
        return _cart_reply("That payment is being refunded -- no order was placed. Your cart's still here.")
    if status.get("is_terminal_failure") or state_status == "failed":
        return _cart_reply("That payment didn't go through -- your cart's still here, try checking out again.")

    confirm = food_mcp.confirm_order(
        order.get("order_id"), address_id, order.get("lat"), order.get("lng"), cart_id=order.get("cart_id"),
    )
    debug("cart_agent", f"checkout: still pending at the poll deadline, confirm_order -> {confirm!r}")
    return _cart_reply(
        f"Your payment is still processing (order {order.get('order_id')}) -- "
        "check the Swiggy app for the latest status; I'll stop waiting here."
    )


def _order_placed_reply(order: dict[str, Any], method_label: str, confirm: dict[str, Any] | None = None) -> dict:
    order_id = (confirm or {}).get("order_id") or order.get("order_id")
    lines = [f"Order placed! ({method_label})", f"Order ID: {order_id}"]
    if order.get("total_amount") is not None:
        lines.append(f"Total: ₹{order['total_amount']}")
    if order.get("estimated_delivery"):
        lines.append(f"Estimated delivery: {order['estimated_delivery']}")
    order_status = (confirm or {}).get("order_status") or order.get("swiggy_status")
    if order_status:
        lines.append(f"Status: {order_status}")
    return _cart_reply("\n".join(lines), restaurant=None)
