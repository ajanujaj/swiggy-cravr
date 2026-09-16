"""search_menu (direct dish-name search) and search_restaurants (the
restaurant-first fallback), plus the local filtering both funnel into."""

from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first
from .menu import _IMAGE_REF_RE, _normalize_item, _parse_menu_items

# EATRIGHT-flavored tags from RecommendAgent.KNOWN_TAGS (vegan/vegetarian
# excluded -- those are dietary/cuisine, not "healthy eating").
_EATRIGHT_TAGS = {"low-calorie", "high-protein", "low-carb", "gluten-free", "sugar-free-option", "organic"}


def _infer_collection(filters: dict[str, Any]) -> str | None:
    """Optional `collection` scope for search_restaurants. No BOLT mapping --
    nothing in Cravr signals "fast delivery" yet."""
    tags = {t.strip() for t in str(filters.get("tags") or "").split(",") if t.strip()}
    if tags & _EATRIGHT_TAGS:
        return "EATRIGHT"
    max_price = filters.get("max_price_inr")
    if isinstance(max_price, (int, float)) and max_price <= 99:
        return "STORE_99"
    return None


# search_menu's real response is plain text, not JSON (confirmed from a live
# CRAVR_DEBUG log) -- a numbered list like:
#   Found 10 menu items for "Bolognese":
#   1. Spaghetti Bolognese Pasta — ₹625 | 4.5★ | Fat Lulu's Cafe & Bar (restaurantId: 30446) (ID: 35627636) [has addons]
#   6. Sicilian Four Cheese — ₹545 | Veg | Fat Lulu's Cafe & Bar (restaurantId: 30446) (ID: 55029917) [has addons]
# The "|"-separated middle segment holds 0-2 optional attributes (a "Veg"/
# "Non-veg" tag, a "N.N★" rating) before the restaurant name, which is
# always last.
_SEARCH_MENU_ITEM_RE = re.compile(
    r"^\d+\.\s*(?P<name>.+?)\s*—\s*₹(?P<price>[\d,]+(?:\.\d+)?)\s*\|\s*(?P<rest>.+?)\s*"
    r"\(restaurantId:\s*(?P<restaurant_id>[^)]+)\)\s*\(ID:\s*(?P<id>[^)]+)\)"
    r"(?:\s*\[(?P<tags>[^\]]+)\])?\s*$",
    re.MULTILINE,
)
_RATING_RE = re.compile(r"^\d+(?:\.\d+)?★$")


def _parse_search_menu_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in _SEARCH_MENU_ITEM_RE.finditer(text):
        rest_parts = [p.strip() for p in m.group("rest").split("|") if p.strip()]
        restaurant_name = rest_parts[-1] if rest_parts else ""
        rating: str | None = None
        dietary_type = ""
        for part in rest_parts[:-1]:
            if _RATING_RE.match(part):
                rating = part.rstrip("★")
            elif part.lower() in ("veg", "non-veg", "nonveg", "non veg"):
                dietary_type = "Veg" if part.lower() == "veg" else "Non-Veg"

        try:
            price_inr: float | int = float(m.group("price").replace(",", ""))
        except ValueError:
            price_inr = 0

        items.append({
            "item_name": m.group("name").strip(),
            "menu_item_id": m.group("id").strip(),
            "restaurant_name": restaurant_name,
            "restaurant_id": m.group("restaurant_id"),
            "cuisine": "",
            "price_inr": price_inr,
            "rating": rating,
            "tags": m.group("tags") or "",
            "dietary_type": dietary_type,
        })
    return items


# A *targeted* search_menu call (item name + restaurantIdOfAddedItem, per
# Swiggy's own docs: "To order an item, use search_menu with the item name
# and restaurantId to get full customization details") returns a totally
# different text shape than the broad/unscoped one _SEARCH_MENU_ITEM_RE
# handles -- confirmed via a live CRAVR_DEBUG probe. No restaurantId in
# the header (it's already scoped to one restaurant), rating shown as
# "4.8★ (53)" (review count in parens, no "|" before it), and each item is
# followed by one or more indented "   Addons (<group name>): [...]" lines
# instead of a trailing "[has addons]" tag -- each bracketed entry is
# "<choice name> ₹<price> (group:<group_id>, choice:<choice_id>)".
_DETAILED_ITEM_HEADER_RE = re.compile(
    # Same overall "N. name — ₹price | ... (ID: X)" shape as the broad-search
    # format above, which is why a plain match on this alone false-matched
    # broad-format lines that happen to lack a trailing "[has addons]" tag
    # (confirmed against the real "Hummus & pita" broad-search fixture in
    # tests/test_food_mcp.py). The real, reliable difference: broad-format
    # lines always carry "(restaurantId: X)" before "(ID: Y)" (the targeted
    # shape omits it -- it's already scoped to one restaurant), so a
    # negative lookahead on that is used to tell the two shapes apart
    # rather than relying on the addon-tag text, which isn't always present.
    r"^\d+\.[ \t]*(?P<name>.+?)[ \t]*—[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?)[ \t]*\|[ \t]*"
    r"(?P<rest>(?:(?!\(restaurantId:).)+?)[ \t]*"
    r"\(ID:[ \t]*(?P<id>[^)]+)\)[ \t]*$",
    re.MULTILINE,
)
# "Addons (...)" and "Variants (...)" lines share the same overall shape
# but are NOT interchangeable: confirmed live (logger.txt, "Korean Honey
# Gochujang Fried Chicken Burger", restaurant of ID 187075710) a variant
# group ("Variants (Choice of Cheese): [With Cheese ... (group:X,
# var:Y), ...]") uses "var:" for its id token, not "choice:" -- an earlier
# version of this parser only recognised "Addons (...)" lines at all, so
# every Variants group for every item was silently dropped rather than
# just mis-parsed. `kind` distinguishes the two so callers can tell a real
# size/base choice (variants -- changes what the item IS, no safe default)
# from an optional upsell (addons -- see CartAgent's attempt_addon_only).
_DETAILED_ADDON_GROUP_RE = re.compile(
    r"^[ \t]*(?P<kind>Addons|Variants) \((?P<group_name>[^)]+)\):[ \t]*\[(?P<body>.*)\][ \t]*$",
    re.MULTILINE,
)
_DETAILED_ADDON_CHOICE_RE = re.compile(
    r"(?P<name>.+?)[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?)[ \t]*"
    r"\(group:(?P<group_id>\d+),[ \t]*(?:choice|var):(?P<choice_id>\d+)\)"
)


def _parse_detailed_search_menu_items(text: str) -> dict[str, dict[str, Any]]:
    """Parse the targeted-lookup shape above into
    {menu_item_id: {"item_name", "price_inr",
        "addon_groups": [{"group_id", "group_name", "choices": [...]}],
        "variant_groups": [{"group_id", "group_name", "choices": [...]}],
    }}. addon_groups are optional upsells (see CartAgent's
    attempt_addon_only); variant_groups are real size/base-item choices
    (e.g. "Choice of Cheese") that change what the item IS -- kept
    separate rather than merged, since only the addon kind currently gets
    a safe default. Returns {} if the text does not match this shape at
    all (e.g. it was actually the broad-search shape instead) -- callers
    should treat an empty dict as "nothing usable", not as "this item has
    no addons/variants".
    """
    headers = list(_DETAILED_ITEM_HEADER_RE.finditer(text))
    items: dict[str, dict[str, Any]] = {}
    for idx, m in enumerate(headers):
        item_id = m.group("id").strip()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[m.end():block_end]

        try:
            price_inr: float | int = float(m.group("price").replace(",", ""))
        except ValueError:
            price_inr = 0

        addon_groups = []
        variant_groups = []
        for gm in _DETAILED_ADDON_GROUP_RE.finditer(block):
            choices = []
            for cm in _DETAILED_ADDON_CHOICE_RE.finditer(gm.group("body")):
                try:
                    choice_price: float | int = float(cm.group("price").replace(",", ""))
                except ValueError:
                    choice_price = 0
                choices.append({
                    "id": cm.group("choice_id"),
                    # finditer matches choices back-to-back separated by ", " --
                    # every match after the first starts with that leftover
                    # ", " (the comma belongs to the previous choice's
                    # separator, not this choice's name), so strip comma+space
                    # specifically rather than plain .strip().
                    "name": cm.group("name").strip().strip(", ").strip(),
                    "price_inr": choice_price,
                    "group_id": cm.group("group_id"),
                })
            # The group header line ("Addons (<name>): [...]" or "Variants
            # (<name>): [...]") never carries a group_id itself -- only the
            # choices inside it do, via "(group:X, choice:Y)"/"(group:X,
            # var:Y)". Every choice in one group shares the same group_id,
            # so promote the first choice's id to the group level.
            group_record = {
                "group_id": choices[0]["group_id"] if choices else None,
                "group_name": gm.group("group_name").strip(),
                "choices": choices,
            }
            (variant_groups if gm.group("kind") == "Variants" else addon_groups).append(group_record)
        items[item_id] = {
            "item_name": m.group("name").strip(),
            "price_inr": price_inr,
            "addon_groups": addon_groups,
            "variant_groups": variant_groups,
        }
    return items


def search_menu_items(
    query: str, address_id: str | None, veg_only: bool = False, offset: int = 0,
    restaurant_id: str | None = None,
) -> dict[str, Any]:
    """Search dishes directly by name across restaurants via Swiggy's
    search_menu tool -- more direct than search_restaurants +
    get_restaurant_menu for a specific craving ("sushi", "salad", "bolognese").

    The response is a numbered plain-text list (see `_parse_search_menu_items`),
    same style as get_restaurant_menu/get_addresses -- not the JSON shape the
    docs alone suggested (an earlier version of this function assumed JSON and
    silently discarded every real result as a result). No pagination info is
    exposed in the text form, so has_more/next_offset are always False/None
    here; every raw response is still logged via `debug()` so a future format
    change shows up immediately in CRAVR_DEBUG=1 output.

    Returns {"items": [...normalized dish records...], "has_more": bool,
    "next_offset": int | None}.

    `restaurant_id` maps to the tool's own `restaurantIdOfAddedItem` param
    (per Swiggy's docs -- confirmed there is no plain `restaurantId`), which
    scopes the search to one restaurant; Swiggy's own docs say this
    item-name+restaurant combo is how to "get full customization details"
    (variants/variantsV2/addons) for one specific dish. CONFIRMED live
    (captured in logger.txt, item "Falafel Hummus Pita Pocket", restaurant
    300897): a targeted call like this returns real, structured addon
    group/choice ids (group:NNNN, choice:NNNN) -- but in a THIRD text
    shape, different from both the broad-search line format and the JSON
    the docs describe. See `_parse_detailed_search_menu_items` for that
    shape's parser; its output shows up here under `detailed_items` when
    the broad-format parser finds nothing. No example with `variants`/
    `variantsV2` (as opposed to `addons`) has been seen in this shape yet,
    and no min/maxAddons-style "is this group mandatory" field is exposed
    in it either -- only name/price/ids.
    """
    arguments: dict[str, Any] = {"addressId": address_id, "query": query, "offset": offset}
    if veg_only:
        arguments["vegFilter"] = 1
    if restaurant_id:
        arguments["restaurantIdOfAddedItem"] = restaurant_id

    try:
        raw = _call_tool("search_menu", arguments)
    except Exception as exc:
        debug("food_mcp", f"search_menu FAILED: {exc}")
        return {"items": [], "has_more": False, "next_offset": None}

    if isinstance(raw, str):
        parsed_items = _parse_search_menu_items(raw)
        result: dict[str, Any] = {"items": parsed_items, "has_more": False, "next_offset": None}
        if not parsed_items:
            # The broad-search line format didn't match -- this may be the
            # other real shape a targeted (restaurant_id-scoped) query
            # returns instead (see _parse_detailed_search_menu_items). Only
            # populated when it actually found something, so callers can
            # tell "nothing usable" (both empty) from "detailed shape only"
            # (items empty, detailed_items populated) without re-parsing raw
            # text themselves.
            detailed = _parse_detailed_search_menu_items(raw)
            if detailed:
                result["detailed_items"] = detailed
        return result

    if not isinstance(raw, dict):
        return {"items": [], "has_more": False, "next_offset": None}

    # Defensive fallback in case Swiggy ever actually returns the JSON shape
    # the docs describe.
    items: list[dict[str, Any]] = []
    for entry in raw.get("items") or []:
        if not isinstance(entry, dict):
            continue
        restaurant = entry.get("restaurant") if isinstance(entry.get("restaurant"), dict) else {}
        restaurant_name = _first(entry, "restaurantName") or _first(restaurant, "name", "restaurantName", default="")
        cuisine = _first(entry, "cuisine") or _first(restaurant, "cuisine", "cuisines")
        if isinstance(cuisine, list):
            cuisine = ", ".join(str(c) for c in cuisine)
        name = _first(entry, "name", "itemName", "dishName", default="")
        if not name:
            continue
        items.append({
            "item_name": name,
            "menu_item_id": _first(entry, "id", "itemId", "menuItemId"),
            "restaurant_name": restaurant_name or "",
            "restaurant_id": _first(entry, "restaurantId") or _first(restaurant, "id", "restaurantId"),
            "cuisine": cuisine or "",
            "price_inr": _first(entry, "price", "priceInr", "price_inr", default=0),
            "rating": _first(entry, "rating", "restaurantRating"),
            "tags": _first(entry, "tags", default="") or "",
            "dietary_type": "Veg" if _first(entry, "isVeg", "veg") else "",
        })

    return {
        "items": items,
        "has_more": bool(raw.get("hasMore")),
        "next_offset": raw.get("nextOffset"),
    }


def search_dishes(filters: dict[str, Any], address_id: str | None, limit_restaurants: int = 5) -> list[dict[str, Any]]:
    """search_menu (direct dish-name search) when a specific craving is known --
    e.g. "sushi" from the category/type funnel in intentAgent.py -- else
    search_restaurants -> get_restaurant_menu -> flatten -> filter locally
    (Swiggy's tools don't filter by dish attributes beyond a name/cuisine query).
    """
    craving = filters.get("craving")
    if craving:
        veg_only = filters.get("dietary_type") == "Veg"
        result = search_menu_items(str(craving), address_id, veg_only=veg_only)
        if result["items"]:
            # dietary_type isn't re-checked locally here -- vegFilter above already asked
            # Swiggy to enforce it server-side, and search_menu_items' own veg/isVeg field
            # name is an unverified guess (see its docstring); re-applying a strict local
            # check on a field that might simply be absent could zero out every result.
            local_filters = {k: v for k, v in filters.items() if k != "dietary_type"}
            filtered = _apply_filters(result["items"], local_filters)
            if filtered:
                return filtered
        debug("food_mcp", f"search_menu found nothing usable for {craving!r} -- falling back to search_restaurants")

    query = filters.get("cuisine") or filters.get("tags") or filters.get("dietary_type") or "restaurants"
    arguments: dict[str, Any] = {"addressId": address_id, "query": query}
    collection = _infer_collection(filters)
    if collection:
        arguments["collection"] = collection

    try:
        raw = _call_tool("search_restaurants", arguments)
    except Exception as exc:
        debug("food_mcp", f"search_restaurants FAILED: {exc}")
        return []

    restaurants = raw.get("restaurants") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not restaurants:
        return []

    open_restaurants = [
        r for r in restaurants
        if isinstance(r, dict) and _first(r, "availabilityStatus", default="OPEN") == "OPEN"
    ][:limit_restaurants]

    dishes: list[dict[str, Any]] = []
    for restaurant in open_restaurants:
        restaurant_id = _first(restaurant, "restaurantId", "id")
        restaurant_name = _first(restaurant, "name", "restaurantName", default="")
        if not restaurant_id:
            continue
        try:
            # addressId is required here too, same as search_restaurants.
            menu_raw = _call_tool(
                "get_restaurant_menu", {"restaurantId": restaurant_id, "addressId": address_id},
                context=f"for {restaurant_id}",
            )
        except Exception as exc:
            debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
            continue
        if isinstance(menu_raw, str):
            menu_raw = _IMAGE_REF_RE.sub("", menu_raw)
        # "cuisines" is a list, not singular "cuisine" -- join for the filter's substring match.
        restaurant_cuisine = _first(restaurant, "cuisine", "cuisines")
        if isinstance(restaurant_cuisine, list):
            restaurant_cuisine = ", ".join(str(c) for c in restaurant_cuisine)

        if isinstance(menu_raw, str):
            dishes.extend(_parse_menu_items(menu_raw, restaurant_name, restaurant_cuisine, restaurant_id))
        else:
            # Defensive fallback if the response is ever real JSON instead of text.
            raw_items = menu_raw.get("items") if isinstance(menu_raw, dict) else (menu_raw if isinstance(menu_raw, list) else [])
            for item in raw_items or []:
                dishes.append(_normalize_item(item, restaurant_name, restaurant_cuisine, restaurant_id))

    dishes = _apply_filters(dishes, filters)
    if craving:
        # This is the untargeted fallback (search_menu found nothing usable) --
        # without a craving-name filter it would hand back a restaurant's whole
        # menu regardless of what the user actually asked for.
        dishes = [d for d in dishes if _matches_craving(str(d.get("item_name") or ""), str(craving))]
    return dishes


def _apply_filters(dishes: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    dietary_type = filters.get("dietary_type")
    max_price = filters.get("max_price_inr")
    cuisine = (filters.get("cuisine") or "").lower()

    def keep(d: dict[str, Any]) -> bool:
        if dietary_type and str(d.get("dietary_type") or "").lower() != dietary_type.lower():
            return False
        if max_price is not None and isinstance(d.get("price_inr"), (int, float)) and d["price_inr"] > max_price:
            return False
        if cuisine and cuisine not in str(d.get("cuisine") or "").lower():
            return False
        return True

    return [d for d in dishes if keep(d)]


_CRAVING_STOPWORDS = {"a", "an", "and", "the", "or", "with", "some", "of", "some", "please"}


def _matches_craving(item_name: str, craving: str) -> bool:
    """Loose substring/token match between a dish name and what the user actually
    asked for. Only used on the search_restaurants -> get_restaurant_menu fallback:
    that path searches for RESTAURANTS by cuisine/tag text and, once it has one,
    returns that restaurant's *entire* menu -- with no dish-level filtering at all.
    Without this, e.g. a "Mac and cheese" craving that falls back to a "comfort
    food" restaurant search would return literally every dish on that restaurant's
    menu (kebabs, biryani, dal makhani, ...), none of them mac and cheese."""
    name = (item_name or "").lower()
    craving_l = (craving or "").lower().strip()
    if not craving_l:
        return True
    if craving_l in name or name in craving_l:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", craving_l) if w not in _CRAVING_STOPWORDS and len(w) > 2]
    if not words:
        return True
    return any(w in name for w in words)
