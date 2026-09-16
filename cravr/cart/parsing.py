"""Text parsing/formatting for the cart's own state: pulling
{menu_item_id, quantity} pairs back out of a real get_food_cart/
update_food_cart plain-text response, categorizing a menu item into a
coarse bucket for "add more" pairing, and rendering the cart for display.
"""

from __future__ import annotations

import re
from typing import Any

# Best-effort extraction of {menu_item_id, quantity} pairs from a real
# get_food_cart/update_food_cart plain-text item list, e.g.
# "  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)". Swiggy's cart
# responses are plain text with no JSON items array for the real tool (see
# food_mcp.cart._strip_widget_disclaimer's docstring), so this is the only
# way to know what's already in the cart before merging in a newly-added
# item. Only the single-item, quantity-omitted format has been observed
# live; an "x<N>" quantity suffix is supported defensively but unverified
# against a real multi-quantity response.
_CART_LINE_RE = re.compile(
    r"^\s*-\s*.+?(?:\s+x(?P<qty>\d+))?\s*[—-]\s*₹[\d,.]+\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)


def _parse_existing_cart_items(cart_message: str) -> list[dict[str, Any]]:
    """See _CART_LINE_RE. Returns [{"menu_item_id": ..., "quantity": ...}]."""
    items = []
    for match in _CART_LINE_RE.finditer(cart_message or ""):
        items.append({
            "menu_item_id": match.group("id").strip(),
            "quantity": int(match.group("qty")) if match.group("qty") else 1,
        })
    return items


# Menu category header text -> a coarse bucket, for pairing "add more"
# suggestions with what's already in the cart. Matched against
# get_restaurant_menu's "## Category" headers (see
# food_mcp.menu._parse_menu_items' `tags` field), not free text -- a
# smaller, more literal vocabulary than RecommendAgent's intent-parsing
# cues is enough for menu section names.
_DRINK_CUES = ("beverage", "drink", "juice", "shake", "lassi", "soda", "mocktail", "smoothie")
_DESSERT_CUES = ("dessert", "sweet", "ice cream", "cake", "pastry", "pudding")
_SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "add-on", "addon")

# Which buckets to prefer suggesting, given what's already in the cart --
# a main course pairs with a drink/dessert/side, not five more mains.
_COMPLEMENT_ORDER = {
    "main": ["drink", "dessert", "side"],
    "side": ["main", "drink"],
    "drink": ["main", "dessert"],
    "dessert": ["main", "drink"],
}


def _menu_bucket(category_text: str) -> str:
    """Coarse bucket (main/side/drink/dessert) for one menu item's category
    text, via _DRINK_CUES/_DESSERT_CUES/_SIDE_CUES. Defaults to "main" --
    that's the safe default for an uncategorised or genuinely-main item,
    and also what makes an empty/unknown bucket fall back to suggesting
    drinks/desserts/sides rather than more unclassified items."""
    haystack = (category_text or "").lower()
    if any(cue in haystack for cue in _DRINK_CUES):
        return "drink"
    if any(cue in haystack for cue in _DESSERT_CUES):
        return "dessert"
    if any(cue in haystack for cue in _SIDE_CUES):
        return "side"
    return "main"


def _format_cart(cart: dict[str, Any]) -> str:
    items = cart.get("items") or []
    if items:
        lines = [
            f"  - {i.get('item_name')} x{i.get('quantity', 1)} — ₹{i.get('total', i.get('final_price', 0))}"
            for i in items
        ]
        body = "Here's your cart:\n" + "\n".join(lines)
        pricing = cart.get("pricing") or {}
        to_pay = pricing.get("to_pay")
        if to_pay is not None:
            body += f"\nTotal to pay: ₹{to_pay}"
        coupon = cart.get("coupon_applied")
        if coupon:
            body += f"\nCoupon applied: {coupon} (-₹{cart.get('coupon_discount', 0)})"
        return body
    # No structured items -- the real get_food_cart returns plain text with
    # no itemized breakdown (Swiggy expects a widget to show that), so its
    # own status message ("Cart is empty.", etc.) is all there is to show.
    return cart.get("message") or "Your cart is empty."
