from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug

_CART_TOPIC_RE = re.compile(
    r"\bcart\b|\bcoupons?\b|\bpromo\s*codes?\b|\bdiscounts?\b|"
    r"\border it\b|\border this\b|\border that\b|\badd it\b|\badd this\b|\badd that\b",
    re.IGNORECASE,
)
_COUPON_APPLY_RE = re.compile(
    r"\b(?:apply|use|redeem)\s+(?:coupon\s+|code\s+|promo\s+)?([A-Za-z0-9][A-Za-z0-9-]{2,19})\b",
    re.IGNORECASE,
)
_CLEAR_RE = re.compile(r"\b(clear|empty|flush)\b.*\bcart\b", re.IGNORECASE)
_ADD_RE = re.compile(
    r"\badd\b.*\bcart\b|\border (it|this|that)\b|\bput (it|this|that) in (my |the )?cart\b|"
    r"\badd (it|this|that)\b",
    re.IGNORECASE,
)
_VIEW_RE = re.compile(
    r"\b(view|show|see|check)\b.*\bcart\b|\bwhat'?s in (my |the )?cart\b|^\s*(my )?cart\s*$",
    re.IGNORECASE,
)
_COUPON_LIST_RE = re.compile(r"\bcoupons?\b|\boffers?\b|\bdiscounts?\b|\bdeals?\b", re.IGNORECASE)
_QTY_RE = re.compile(r"\b(\d+)\b")

CART_ADD_MORE_VALUE = "__cart_add_more__"
CART_PROCEED_VALUE = "__cart_proceed__"
CART_START_OVER_VALUE = "__cart_start_over__"
CART_CLEAR_AND_ADD_VALUE = "__cart_clear_and_add__"
CART_KEEP_VALUE = "__cart_keep__"
CART_UPI_PAID_VALUE = "__cart_upi_paid__"
CART_UPI_NOT_PAID_VALUE = "__cart_upi_not_paid__"
_NEXT_STEP_SENTINELS = {
    CART_ADD_MORE_VALUE, CART_PROCEED_VALUE, CART_START_OVER_VALUE,
    CART_CLEAR_AND_ADD_VALUE, CART_KEEP_VALUE, CART_UPI_PAID_VALUE, CART_UPI_NOT_PAID_VALUE,
}

_CONFLICT_OPTIONS = [
    {"value": CART_CLEAR_AND_ADD_VALUE, "label": "Clear my cart and add this"},
    {"value": CART_KEEP_VALUE, "label": "Keep my cart as it is"},
]

_NEXT_STEP_OPTIONS = [
    {"value": CART_ADD_MORE_VALUE, "label": "Add more from this restaurant"},
    {"value": CART_PROCEED_VALUE, "label": "Proceed"},
    {"value": CART_START_OVER_VALUE, "label": "Browse other restaurants"},
]

_UPI_CONFIRM_OPTIONS = [
    {"value": CART_UPI_PAID_VALUE, "label": "I've paid"},
    {"value": CART_UPI_NOT_PAID_VALUE, "label": "Cancel — I haven't paid"},
]


def is_cart_request(text: str) -> bool:
    t = (text or "").strip()
    if t in _NEXT_STEP_SENTINELS:
        return True
    return bool(_CART_TOPIC_RE.search(t) or _COUPON_APPLY_RE.search(t))


def _classify_cart_action(text: str) -> tuple[str, Any]:
    t = (text or "").strip()
    if t == CART_ADD_MORE_VALUE:
        return "suggest_more", None
    if t == CART_PROCEED_VALUE:
        return "checkout", None
    if t == CART_START_OVER_VALUE:
        return "start_over", None
    if t == CART_CLEAR_AND_ADD_VALUE:
        return "clear_and_add", None
    if t == CART_KEEP_VALUE:
        return "keep_cart", None
    if t == CART_UPI_PAID_VALUE:
        return "upi_paid", None
    if t == CART_UPI_NOT_PAID_VALUE:
        return "upi_not_paid", None
    if _CLEAR_RE.search(t):
        return "clear", None
    coupon_match = _COUPON_APPLY_RE.search(t)
    if coupon_match:
        return "apply_coupon", coupon_match.group(1)
    if _ADD_RE.search(t):
        qty_match = _QTY_RE.search(t)
        return "add", int(qty_match.group(1)) if qty_match else 1
    if _COUPON_LIST_RE.search(t):
        return "coupons", None
    if _VIEW_RE.search(t):
        return "view", None
    return "view", None


def _log_action(action: str, arg: Any) -> None:
    debug("cart_agent", f"classified action={action!r} arg={arg!r}")
