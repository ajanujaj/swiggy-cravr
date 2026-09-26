"""Deterministic keyword routing for the cart agent: whether a message is
cart-topic at all (`is_cart_request`, graph.py's entry gate), and which
specific cart action it maps to (`_classify_cart_action`) -- same
lightweight-heuristic style as RecommendAgent.is_pure_feedback, no extra
LLM call just to tell "add to cart" from "clear my cart".
"""

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

# Sentinels for the post-add "what next" buttons -- same convention as
# cravr.options.OTHER_VALUE (a fixed, unmistakable value a tap always
# sends, independent of the button's display label).
CART_ADD_MORE_VALUE = "__cart_add_more__"
CART_PROCEED_VALUE = "__cart_proceed__"
CART_START_OVER_VALUE = "__cart_start_over__"
# Offered when the dish being added belongs to a different restaurant than
# what the Swiggy cart already holds (Swiggy carts are single-restaurant).
CART_CLEAR_AND_ADD_VALUE = "__cart_clear_and_add__"
CART_KEEP_VALUE = "__cart_keep__"
_NEXT_STEP_SENTINELS = {
    CART_ADD_MORE_VALUE, CART_PROCEED_VALUE, CART_START_OVER_VALUE,
    CART_CLEAR_AND_ADD_VALUE, CART_KEEP_VALUE,
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


def is_cart_request(text: str) -> bool:
    """Entry-router gate (graph.py) -- only messages with a strong cart/
    coupon signal get routed here at all. Also catches a bare "apply
    SAVE20"/"use FREESHIP" with no "coupon"/"cart" word in it, since
    `_classify_cart_action` supports that phrasing too. The post-add "what
    next" buttons send an exact sentinel (see `_NEXT_STEP_SENTINELS`) that
    wouldn't otherwise match any of the natural-language patterns below."""
    t = (text or "").strip()
    if t in _NEXT_STEP_SENTINELS:
        return True
    return bool(_CART_TOPIC_RE.search(t) or _COUPON_APPLY_RE.search(t))


def _classify_cart_action(text: str) -> tuple[str, Any]:
    t = (text or "").strip()
    if t == CART_ADD_MORE_VALUE:
        return "suggest_more", None
    if t == CART_PROCEED_VALUE:
        return "proceed_placeholder", None
    if t == CART_START_OVER_VALUE:
        return "start_over", None
    if t == CART_CLEAR_AND_ADD_VALUE:
        return "clear_and_add", None
    if t == CART_KEEP_VALUE:
        return "keep_cart", None
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
    # is_cart_request already confirmed cart-topic vocabulary; default to
    # showing the cart rather than silently doing nothing.
    return "view", None


def _log_action(action: str, arg: Any) -> None:
    debug("cart_agent", f"classified action={action!r} arg={arg!r}")
