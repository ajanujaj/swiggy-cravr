"""Cart agent facade: view/add/clear the Swiggy cart and manage coupons.

The actual implementation now lives split across this package (routing.py,
parsing.py, restaurant_context.py, guards.py, display.py, add_flow.py,
flows.py, node.py) -- this module just re-exports the names other code in
the repo imports from `cravr.cart.CartAgent` (RecommendAgent's pick-to-cart
handoff, tests/test_cart_agent.py's patch() targets, this package's own
__init__.py) so none of those call sites had to change.

`food_mcp` is re-exported (not just used internally) because
tests/test_cart_agent.py patches `cravr.cart.CartAgent.food_mcp.get_cart`
and friends -- that works because `food_mcp` here is the SAME module
object every sub-file imports via `from .. import food_mcp`, so patching
an attribute on it here mutates the one object everyone looks it up on.
"""

from __future__ import annotations

from .. import food_mcp
from .add_flow import add_dish_to_cart
from .node import cart_agent_node
from .restaurant_context import _current_restaurant_id, _current_restaurant_name
from .routing import (
    CART_ADD_MORE_VALUE,
    CART_PROCEED_VALUE,
    CART_START_OVER_VALUE,
    CART_CLEAR_AND_ADD_VALUE,
    CART_KEEP_VALUE,
    _CONFLICT_OPTIONS,
    _NEXT_STEP_OPTIONS,
    _classify_cart_action,
    is_cart_request,
)

__all__ = [
    "food_mcp",
    "add_dish_to_cart",
    "cart_agent_node",
    "_current_restaurant_id",
    "_current_restaurant_name",
    "CART_ADD_MORE_VALUE",
    "CART_PROCEED_VALUE",
    "CART_START_OVER_VALUE",
    "CART_CLEAR_AND_ADD_VALUE",
    "CART_KEEP_VALUE",
    "_NEXT_STEP_OPTIONS",
    "_classify_cart_action",
    "is_cart_request",
]
