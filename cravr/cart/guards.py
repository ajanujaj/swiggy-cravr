"""Guards against adding an item Cravr can't actually complete: one that
needs a variant/add-on choice Swiggy's real update_food_cart silently
rejects without, and (in add_flow.py) one from a different restaurant
than what's already in the cart.
"""

from __future__ import annotations

from typing import Any

from .parsing import _COMPLEMENT_ORDER, _menu_bucket


def _requires_customization(tags: str) -> bool:
    """True if get_restaurant_menu flagged this item "has addons"/"has
    variants" (food_mcp.menu._parse_menu_items folds that straight into
    the item's `tags` field). Confirmed via a live CRAVR_DEBUG log for
    restaurant "Mister Edamame": suggesting "Manchow Soup" and "Sichuan
    Hot & Sour Soup" (both tagged "has addons") and picking either one made
    the real update_food_cart respond 'Cart updated.\nCart is empty.' --
    the WHOLE merged cart update was rejected, not just that one item --
    while the same restaurant's addon-free items (e.g. the cart's own
    "Veg Stir Fried Udon Noodles") added fine. Matches the residual risk
    flagged in food_mcp.cart.update_cart's docstring: Cravr has no UI to
    capture a required variant/add-on choice, so these items are excluded
    from "add more" suggestions entirely -- not just deprioritised --
    rather than let Cravr silently fail the add and then lie to the user
    that it worked. (add_flow.py's add_dish_to_cart no longer refuses
    EVERY such item outright -- see its attempt_addon_only -- but this
    check is still the first gate for both paths.)"""
    haystack = (tags or "").lower()
    return "has addons" in haystack or "has variants" in haystack


def _customization_refusal(item_name: str) -> str:
    """The message add_flow.py falls back to whenever an item needing a
    variant/add-on choice can't be added -- both for the "has variants"
    case (refused outright, no safe default exists) and for an addon-only
    attempt that didn't verifiably land. Centralized so the exact wording
    can't drift between the two call sites; deliberately never prefixed
    with a caller's `lead_in` (see add_flow.py's own comment on why)."""
    return (
        f"{item_name or 'That item'} needs a variant/add-on choice I can't set for you yet, "
        "so I haven't added it to your cart -- pick something else, or add it with add-ons directly in the Swiggy app."
    )


def _suggest_complementary_items(
    menu_items: list[dict[str, Any]],
    cart_item_id: str | None,
    exclude_ids: set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Up to `limit` items from `menu_items` that pair with what's already
    in the cart: look up the cart item's own bucket from the freshly
    fetched menu (its category header is the only reliable source -- the
    dish record captured when it was originally picked has no category
    info, see food_mcp.search._parse_search_menu_items), then prefer
    OTHER buckets per `_COMPLEMENT_ORDER`. Tops up with whatever's left
    (any bucket) if the preferred ones don't reach `limit` -- a
    preference, not a hard filter, so "add more" never comes back
    emptier than it has to. Items needing a variant/add-on choice ARE
    hard-filtered out first (see `_requires_customization`) -- unlike the
    bucket preference, that one is not negotiable, since Cravr can't
    supply the required choice and a bare add silently fails against the
    real API."""
    by_id = {str(i.get("menu_item_id")): i for i in menu_items if i.get("menu_item_id")}
    cart_item = by_id.get(str(cart_item_id)) if cart_item_id else None
    cart_bucket = _menu_bucket((cart_item or {}).get("tags") or "")
    preferred_buckets = _COMPLEMENT_ORDER.get(cart_bucket, ["drink", "dessert", "side"])

    # Hard exclusion, not a bucket preference: an item needing a variant/
    # add-on choice must never be suggested -- see _requires_customization.
    candidates = [
        i for i in menu_items
        if str(i.get("menu_item_id")) not in exclude_ids
        and not _requires_customization(i.get("tags") or "")
    ]
    bucketed: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        bucketed.setdefault(_menu_bucket(item.get("tags") or ""), []).append(item)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _take(items: list[dict[str, Any]]) -> None:
        for item in items:
            item_id = str(item.get("menu_item_id"))
            if item_id in seen:
                continue
            seen.add(item_id)
            ordered.append(item)

    for bucket in preferred_buckets:
        _take(bucketed.get(bucket, []))
    _take(candidates)
    return ordered[:limit]
