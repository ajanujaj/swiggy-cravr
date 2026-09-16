"""Which restaurant the cart is currently pinned to -- Swiggy carts are
single-restaurant, so both the cross-restaurant guard (add_flow.py) and
the "add more from this restaurant" flow (flows.py) need this."""

from __future__ import annotations

from ..state import CravrState


def _current_restaurant_id(state: CravrState) -> str | None:
    """Cart is single-restaurant on Swiggy -- prefer whatever's already in
    the cart, else fall back to a just-picked dish's restaurant."""
    cart_state = state.get("cart_state") or {}
    restaurant = cart_state.get("restaurant") or {}
    if isinstance(restaurant, dict) and restaurant.get("id"):
        return restaurant.get("id")
    chosen = (state.get("recommendation_state") or {}).get("last_recommendations") or []
    if len(chosen) == 1:
        return chosen[0].get("restaurant_id")
    return None


def _current_restaurant_name(state: CravrState) -> str | None:
    """Same fallback chain as `_current_restaurant_id`, for the name."""
    cart_state = state.get("cart_state") or {}
    restaurant = cart_state.get("restaurant") or {}
    if isinstance(restaurant, dict) and restaurant.get("name"):
        return restaurant.get("name")
    chosen = (state.get("recommendation_state") or {}).get("last_recommendations") or []
    if len(chosen) == 1:
        return chosen[0].get("restaurant_name")
    return None
