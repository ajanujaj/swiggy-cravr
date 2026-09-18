"""Real Swiggy Food MCP adapter -- maps Cravr's agent contracts onto
Swiggy's Food MCP tools (get_addresses, search_restaurants, search_menu,
get_restaurant_menu, get_food_orders, get_food_cart, update_food_cart,
apply_food_coupon, fetch_food_coupons, flush_food_cart -- all
account-scoped via OAuth, no user_id).

Swiggy's own docs (mcp.swiggy.com) describe most of these as returning
JSON, but live testing (CRAVR_DEBUG=1 logs against the real, deployed
tools) has repeatedly shown otherwise: get_addresses, search_menu,
get_restaurant_menu, get_food_cart, update_food_cart, apply_food_coupon,
fetch_food_coupons and flush_food_cart have ALL been confirmed returning
plain, human-readable text instead, at least some of the time. Every
function below still handles the documented JSON shape too (via
`_unwrap`/`_call_tool`), in case a specific deployment or a future Swiggy
update actually returns it -- but the text-mode path is the one that
matters in practice, and is exercised by real captured fixtures in the
tests, not synthetic data.

Package layout: `_shared.py` has generic helpers (JSON/dict unwrapping);
`addresses.py`, `menu.py`, `search.py`, `orders.py` and `cart.py` each
cover one group of Swiggy tools; this file re-exports their public (and a
few historically-referenced private) names so external callers keep doing
`from .. import food_mcp; food_mcp.get_cart(...)` etc. unchanged.
"""

from __future__ import annotations

from typing import Any

from ..client import call_tool_sync
from ..debug_utils import debug
from ._shared import _unwrap


def _call_tool(
    name: str, arguments: dict[str, Any], *, log_request: bool = False, context: str = "",
    return_meta: bool = False,
) -> Any:
    """The one place every food_mcp function calls a Swiggy MCP tool
    through: unwraps the response and logs it (`raw <name> response
    [context]: ...`), optionally logging the outgoing request first too.

    Every submodule below imports this FUNCTION (`from . import
    _call_tool`), not `call_tool_sync` directly -- deliberately, so a test
    can mock the one real network call this whole package makes with a
    single `patch("cravr.food_mcp.call_tool_sync", ...)`, regardless of
    which submodule's function is actually under test (a single CartAgent
    flow often exercises more than one, e.g. get_food_cart AND
    get_restaurant_menu in the same "add more" turn). That only works
    because `_call_tool` is DEFINED here, in this module -- a bare
    reference to `call_tool_sync` inside a function always resolves via
    that function's own defining module's namespace at call time, which is
    exactly what `unittest.mock.patch("cravr.food_mcp.call_tool_sync")`
    targets. If `call_tool_sync` were instead imported separately into
    each submodule, patching this one path would only affect whichever
    submodule happened to import it last, breaking every other one.

    Raises whatever call_tool_sync itself raises -- callers catch that and
    decide their own fallback, since a failure means something different
    per tool (an empty cart vs. an empty search result vs. a hard error),
    so it is deliberately NOT swallowed here.

    `return_meta=True` additionally returns `(raw, meta)`, `meta` being
    `{"is_error", "structured_content", "data"}` off fastmcp's own
    `CallToolResult` (see the diagnostic log line above -- same fields,
    just handed back instead of only logged). Opt-in and defaulted off so
    every existing call site keeps getting just `raw` back unchanged.
    Added for update_food_cart specifically: a live INVALID_ADDON rejection
    (logger.txt, 'Roasted Veggie Pita Pocket', 2026-09-18) came back as
    `status: success` / `'Cart updated.\nCart is empty.'` in the TEXT
    content block, while `structured_content` had the real, explicit
    answer the whole time -- `{'successful': False, 'errorCodes':
    ['INVALID_ADDON'], 'titleMessage': 'Apologies! one or more items in
    your cart are no longer available', ...}`. `_unwrap`/text-mode parsing
    never looked at that field, so `update_cart` had no way to tell "Swiggy
    rejected this specific add-on" apart from "the write silently no-op'd
    for an unknown reason" -- both surfaced identically as the generic
    `_customization_refusal` message.
    """
    if log_request:
        debug("food_mcp", f"{name} request: {arguments!r}")

    result = call_tool_sync(name, arguments)

    # `result` here is fastmcp's own `CallToolResult` dataclass, not the
    # raw MCP-protocol one -- it carries `.content` (the human-text blocks
    # `_unwrap` below reads) but ALSO `.structured_content`, `.data`,
    # `.meta` and `.is_error`, none of which `_unwrap` ever looks at, and
    # which -- until this block -- were never logged either, because
    # `_unwrap` threw them away before `debug()` ever saw the result. That
    # gap matters specifically for update_food_cart: every real call so
    # far has come back `status: success` with message 'Cart updated.

    # Cart is empty.', and the RAW (pre-strip) text additionally contains
    # '...Cart widget is displayed. Wait for the user to review and
    # confirm.' -- wording that implies the write is staged behind some
    # kind of widget/confirmation step rather than committed outright.
    # The MCP protocol's own `CallToolResult` type has a `resultType`
    # field (`"complete" | "input_required"`) for exactly this "not done
    # yet, needs more from the client" case -- but fastmcp's wrapper does
    # NOT surface it (confirmed by reading fastmcp's own
    # client/mixins/tools.py: it copies `content`/`structured_content`/
    # `meta`/`is_error` from the raw result but drops `resultType`
    # entirely), so it can't be recovered from here. `structured_content`/
    # `data`/`meta` ARE passed through by fastmcp though, and have never
    # actually been inspected -- logging them raw so the next live
    # update_food_cart attempt's logger.txt has a chance of showing
    # whatever Swiggy is putting there instead of just the text
    # narrative `_strip_widget_disclaimer` throws away.
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
    update_cart,
)

# Every name below is a re-export for an external caller (`food_mcp.X(...)`
# from another module) or a direct `from cravr.food_mcp import X` (tests,
# mainly) -- not local. Listing them all in __all__ is both the accurate
# public-surface declaration and what tells pyflakes these imports are
# intentional, not dead code.
__all__ = [
    "call_tool_sync",
    "get_saved_address",
    "get_restaurant_menu_items",
    "search_menu_items",
    "search_dishes",
    "get_order_history",
    "get_cart",
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
