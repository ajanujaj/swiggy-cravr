from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.getenv("CRAVR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug(source: str, message: str) -> None:
    if debug_enabled():
        print(f"[debug:{source}] {message}", file=sys.stderr)


def test_mode_enabled() -> bool:
    """CRAVR_TEST_MODE=1 short-circuits the checkout flow (cravr/cart/checkout.py)
    right before place_food_order/confirm_order/check_payment_status -- Swiggy
    has no sandbox/dummy-payment mode (checked against their docs), so every
    real call places a REAL order (COD dispatches an actual delivery, UPI
    moves actual money). With this on, checkout fabricates the same-shaped
    responses locally instead of calling Swiggy, so the whole conversation
    flow (payment-method question, order summary, confirmation message) can
    be exercised safely. Unset (or anything other than 1/true/yes/on) before
    going live -- e.g. on Render, don't set this env var at all."""
    return os.getenv("CRAVR_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
