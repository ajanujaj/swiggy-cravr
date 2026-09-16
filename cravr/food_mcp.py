"""Real Swiggy Food MCP adapter -- maps Cravr's agent contracts onto
Swiggy's Food MCP tools (get_addresses, search_restaurants,
get_restaurant_menu, get_food_orders, all account-scoped via OAuth, no
user_id). Their responses are text, not clean JSON: get_addresses is a
natural-language block; the rest are JSON with trailing human-readable
text appended after it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .client import call_tool_sync
from .debug_utils import debug


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _unwrap(result: Any) -> Any:
    if isinstance(result, (dict, list)):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                return _parse_leading_json(text)
    return result


_JSON_DECODER = json.JSONDecoder()


def _parse_leading_json(text: str) -> Any:
    """Parse a JSON prefix, ignoring trailing text Swiggy appends after it
    (e.g. a UI-widget disclaimer) -- plain json.loads chokes on that with
    "Extra data". Falls back to the raw text if it isn't JSON at all
    (get_addresses' plain-language response)."""
    try:
        value, _end = _JSON_DECODER.raw_decode(text.lstrip())
        return value
    except json.JSONDecodeError:
        return text


_ADDRESS_LINE_RE = re.compile(
    r"^\d+\.\s*\[(?P<label>[^\]]+)\]\s*[^:]*:\s*(?P<address>.+?)\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)
_SUGGESTED_ID_RE = re.compile(r"Suggested/preselected address ID:\s*([^\s.,]+)")

# get_restaurant_menu inlines "[image: <url>]" after almost every line; noise, drop it.
_IMAGE_REF_RE = re.compile(r"\s*\[image:[^\]]*\]")

# get_restaurant_menu's shape: "##"/"###" category headers + dish lines like
# "  - Aloo Paratha ... -- ₹105 | Veg, Bestseller, has addons (ID: 200268099)".
# "###" lines are indented -- the leading "[ \t]*" is required or subcategory
# attribution silently drops.
_MENU_HEADER_RE = re.compile(r"^[ \t]*(#{2,3})\s*(.+?)\s*$", re.MULTILINE)
_MENU_ITEM_RE = re.compile(
    r"^[ \t]*-[ \t]*(?P<name>.+?)(?:[ \t]*—[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?))?"
    r"[ \t]*\|[ \t]*(?P<tags>[^()\n]*?)[ \t]*\(ID:[ \t]*(?P<id>[^)]+)\)[ \t]*$",
    re.MULTILINE,
)
_DIETARY_TAGS = {"veg", "non-veg", "nonveg", "non veg", "egg"}


def _parse_menu_items(
    text: str, restaurant_name: str, restaurant_cuisine: Any, restaurant_id: str | None = None
) -> list[dict[str, Any]]:
    """Parse get_restaurant_menu's text into dish records. Category
    attribution: for each item, take whichever header most recently
    preceded it (## resets subcategory, ### sets it)."""
    headers = [(m.start(), len(m.group(1)), m.group(2)) for m in _MENU_HEADER_RE.finditer(text)]

    def _category_path(pos: int) -> tuple[str, str]:
        category, subcategory = "", ""
        for start, level, title in headers:
            if start >= pos:
                break
            if level == 2:
                category, subcategory = title, ""
            else:
                subcategory = title
        return category, subcategory

    items: list[dict[str, Any]] = []
    for m in _MENU_ITEM_RE.finditer(text):
        raw_tags = [t.strip() for t in m.group("tags").split(",") if t.strip()]
        dietary_tag = next((t for t in raw_tags if t.lower() in _DIETARY_TAGS), "")
        category, subcategory = _category_path(m.start())
        # ";"-joined: _rank_candidates (RecommendAgent.py) splits tags on ";", not ",".
        tags = "; ".join(t for t in (category, subcategory, *raw_tags) if t)

        price_text = m.group("price")
        try:
            price_inr: float | int = float(price_text.replace(",", "")) if price_text else 0
        except ValueError:
            price_inr = 0

        items.append(
            {
                "item_name": m.group("name").strip(),
                "menu_item_id": m.group("id").strip(),
                "restaurant_name": restaurant_name,
                "restaurant_id": restaurant_id,
                "cuisine": restaurant_cuisine or "",
                "price_inr": price_inr,
                "rating": None,
                "tags": tags,
                "dietary_type": "Veg" if dietary_tag.lower() == "veg" else ("Non-Veg" if dietary_tag else ""),
            }
        )
    return items


def get_saved_address() -> dict[str, Any]:
    """-> {"location", "delivery_area", "address_id", "status"}, matching
    ContextAgent.fetch_user_location's contract.

    get_addresses returns a natural-language block, e.g.:
        1. [Baba] Anuj Jain: 307, third floor, ... (ID: abc123)
        Suggested/preselected address ID: abc123. Ask the user to confirm...
    Parsed with regex; uses Swiggy's own suggested ID so this stays ambient
    (no clarification question needed).
    """
    try:
        raw = _unwrap(call_tool_sync("get_addresses", {}))
    except Exception as exc:
        debug("food_mcp", f"get_addresses FAILED ({type(exc).__name__}: {exc})")
        return {"status": "error", "message": f"Unable to retrieve saved address: {exc}"}

    debug("food_mcp", f"raw get_addresses response: {raw!r}")
    text = raw if isinstance(raw, str) else json.dumps(raw)

    addresses = {
        m.group("id"): (m.group("label"), m.group("address"))
        for m in _ADDRESS_LINE_RE.finditer(text)
    }
    if not addresses:
        return {"status": "no_results", "message": "No saved addresses found"}

    suggested = _SUGGESTED_ID_RE.search(text)
    address_id = suggested.group(1) if suggested and suggested.group(1) in addresses else next(iter(addresses))
    label, address_line = addresses[address_id]

    return {
        "status": "success",
        "location": label,
        "delivery_area": address_line,
        "address_id": address_id,
    }


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
    (per docs.md -- confirmed there is no plain `restaurantId`), which
    scopes the search to one restaurant; docs.md's own item-line text says
    this item-name+restaurant combo is how to "get full customization
    details" (variants/variantsV2/addons) for one specific dish -- unverified
    against the real tool, since every response actually seen from it so
    far (broad, unscoped queries) has been a condensed text summary, not
    the structured JSON the docs describe for variants/addons.
    """
    arguments: dict[str, Any] = {"addressId": address_id, "query": query, "offset": offset}
    if veg_only:
        arguments["vegFilter"] = 1
    if restaurant_id:
        arguments["restaurantIdOfAddedItem"] = restaurant_id

    try:
        raw = _unwrap(call_tool_sync("search_menu", arguments))
    except Exception as exc:
        debug("food_mcp", f"search_menu FAILED: {exc}")
        return {"items": [], "has_more": False, "next_offset": None}

    debug("food_mcp", f"raw search_menu response: {raw!r}")

    if isinstance(raw, str):
        return {"items": _parse_search_menu_items(raw), "has_more": False, "next_offset": None}

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
        raw = _unwrap(call_tool_sync("search_restaurants", arguments))
    except Exception as exc:
        debug("food_mcp", f"search_restaurants FAILED: {exc}")
        return []

    debug("food_mcp", f"raw search_restaurants response: {raw!r}")
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
            menu_raw = _unwrap(
                call_tool_sync("get_restaurant_menu", {"restaurantId": restaurant_id, "addressId": address_id})
            )
        except Exception as exc:
            debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
            continue
        if isinstance(menu_raw, str):
            menu_raw = _IMAGE_REF_RE.sub("", menu_raw)
        debug("food_mcp", f"raw get_restaurant_menu response for {restaurant_id}: {menu_raw!r}")
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


def get_restaurant_menu_items(
    restaurant_id: str, address_id: str | None, restaurant_name: str = ""
) -> list[dict[str, Any]]:
    """get_restaurant_menu for ONE already-known restaurant -- e.g. the one
    already in the cart, when cart_agent wants to suggest more from it.
    Unlike `search_dishes`, which loops over several restaurants from a
    fresh search, this fetches exactly one restaurant's full menu. Reuses
    `_parse_menu_items`, the same text parser `search_dishes` already
    relies on for get_restaurant_menu, so category/tags attribution
    (## / ### headers) is identical -- that's what lets a caller work out
    which items are mains vs. sides vs. drinks vs. desserts.
    """
    arguments: dict[str, Any] = {"restaurantId": restaurant_id, "addressId": address_id}
    try:
        raw = _unwrap(call_tool_sync("get_restaurant_menu", arguments))
    except Exception as exc:
        debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
        return []

    if isinstance(raw, str):
        raw = _IMAGE_REF_RE.sub("", raw)
    debug("food_mcp", f"raw get_restaurant_menu response for {restaurant_id}: {raw!r}")

    if isinstance(raw, str):
        return _parse_menu_items(raw, restaurant_name, "", restaurant_id)
    if isinstance(raw, dict):
        raw_items = raw.get("items") if isinstance(raw.get("items"), list) else []
        return [
            _normalize_item(item, restaurant_name, "", restaurant_id)
            for item in raw_items if isinstance(item, dict)
        ]
    return []


def _normalize_item(
    item: dict[str, Any], restaurant_name: str, restaurant_cuisine: Any, restaurant_id: str | None = None
) -> dict[str, Any]:
    is_veg = _first(item, "isVeg", "veg")
    return {
        "item_name": _first(item, "name", "itemName", default="Item"),
        "menu_item_id": _first(item, "id", "itemId", "menuItemId"),
        "restaurant_name": restaurant_name,
        "restaurant_id": restaurant_id or _first(item, "restaurantId"),
        "cuisine": _first(item, "cuisine", default=restaurant_cuisine) or "",
        "price_inr": _first(item, "price", "finalPrice", "priceInr", default=0),
        "rating": _first(item, "rating"),
        "tags": _first(item, "tags", "category", default=""),
        "dietary_type": "Veg" if is_veg in (True, "true", 1) else ("Non-Veg" if is_veg is not None else ""),
    }


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


def get_order_history(address_id: str | None = None) -> dict[str, Any]:
    """-> {"status", "count", "orders"}, matching preference.py's
    fetch_order_history contract.

    addressId is required. If the caller doesn't have one yet (usually
    true -- this runs before context_agent resolves it), resolve it here,
    deliberately uncached (see _resolve_address_id in RecommendAgent.py).
    """
    if not address_id:
        address_id = get_saved_address().get("address_id")
    if not address_id:
        return {"status": "error", "message": "No saved address available to fetch order history"}

    try:
        raw = _unwrap(call_tool_sync("get_food_orders", {"addressId": address_id}))
    except Exception as exc:
        debug("food_mcp", f"get_food_orders FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve order history: {exc}"}

    debug("food_mcp", f"raw get_food_orders response: {raw!r}")
    orders = raw.get("orders") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not orders:
        return {"status": "no_results", "message": "No order history found"}

    normalized = [
        {
            "restaurant_name": _first(o, "restaurantName", "restaurant_name", default=""),
            "cuisine": _first(o, "cuisine", default=""),
            "item_name": _first(o, "itemName", "item_name", default=""),
            "meal_type": _first(o, "mealType", "meal_type", default=""),
            "order_date": _first(o, "orderDate", "order_date", default=""),
            "order_time": _first(o, "orderTime", "order_time", default=""),
            "tags": _first(o, "tags", default=""),
        }
        for o in orders
    ]
    return {"status": "success", "count": len(normalized), "orders": normalized}


# ---------------------------------------------------------------------------
# Cart: get_food_cart / update_food_cart / apply_food_coupon /
# fetch_food_coupons / flush_food_cart. Docs describe a consistent
# {"success", "data", "message"} JSON envelope for all five, unlike
# get_addresses/search_menu (plain text) -- still routed through _unwrap so a
# stray text reply degrades to `_check_success` returning None / an "error"
# status instead of crashing, per the search_menu lesson (docs != reality).
# ---------------------------------------------------------------------------

def _check_success(raw: Any) -> str | None:
    """None when `raw` isn't the documented {"success": false, "error": {...}}
    shape; else the error message to surface."""
    if isinstance(raw, dict) and raw.get("success") is False:
        return _first(raw.get("error") or {}, "message", default="Request failed")
    return None


_DISCLAIMER_MARKERS = (
    "cart widget is displayed",
    "\nnote:",
    "\u26a0",  # warning-sign emoji prefix on "A rich UI widget may be shown..."
)


def _strip_widget_disclaimer(text: str) -> str:
    """Real cart-tool responses are plain text (confirmed via CRAVR_DEBUG=1
    logs, same lesson as search_menu -- docs describe JSON, reality doesn't):
    a user-facing section (status/items, and for a non-empty cart also a
    pricing breakdown -- Item total/Delivery/Taxes & charges/Coupon/TO PAY,
    itself blank-line-separated from the item list), followed by a blank
    line and agent-only meta-instructions about a "cart widget" our chat UI
    doesn't render ("Cart widget is displayed...", "NOTE: ...", the same "A
    rich UI widget may be shown..." boilerplate search_menu/get_restaurant_menu
    append after their JSON).

    A naive "split on the first blank line" truncates the pricing breakdown
    for a non-empty cart, since that response has THREE blank-line-separated
    paragraphs (items, pricing, disclaimer) not two -- confirmed directly in
    a real get_food_cart/update_food_cart response. So instead we search for
    the earliest occurrence of an actual disclaimer marker and keep
    everything before it; only if no marker is found do we fall back to the
    old first-blank-line split, as a defensive last resort.
    """
    lower = text.lower()
    earliest = None
    for marker in _DISCLAIMER_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest].strip()
    return text.split("\n\n", 1)[0].strip()


def get_cart(address_id: str, restaurant_name: str | None = None) -> dict[str, Any]:
    """get_food_cart -> {"status", "cart_id", "restaurant", "items",
    "item_count", "pricing", "coupon_applied", "coupon_discount", "address_id",
    "message"}. items[]: {"menu_item_id","item_name","quantity","total",
    "final_price","in_stock"} -- populated only on the documented-JSON path
    (e.g. the mock server); the real tool returns plain text with no
    itemized breakdown (see `_strip_widget_disclaimer`), so `message` is
    what actually has the human-readable status there.
    """
    arguments: dict[str, Any] = {"addressId": address_id}
    if restaurant_name:
        arguments["restaurantName"] = restaurant_name
    debug("food_mcp", f"get_food_cart request: {arguments!r}")

    try:
        raw = _unwrap(call_tool_sync("get_food_cart", arguments))
    except Exception as exc:
        debug("food_mcp", f"get_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to retrieve cart: {exc}"}

    debug("food_mcp", f"raw get_food_cart response: {raw!r}")

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        result = {
            "status": "empty" if "cart is empty" in message.lower() else "text",
            "cart_id": None, "restaurant": None, "items": [], "item_count": 0,
            "pricing": {}, "coupon_applied": None, "coupon_discount": 0,
            "address_id": address_id, "message": message,
        }
        debug("food_mcp", f"get_food_cart parsed (text mode): status={result['status']!r} message={message!r}")
        return result

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"get_food_cart parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected get_food_cart response"}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    items = [
        {
            "menu_item_id": _first(it, "menu_item_id", "id", default=""),
            "item_name": _first(it, "name", "item_name", default=""),
            "quantity": _first(it, "quantity", default=1),
            "total": _first(it, "total", default=0),
            "final_price": _first(it, "final_price", "total", default=0),
            "in_stock": it.get("in_stock", True),
        }
        for it in (data.get("items") or []) if isinstance(it, dict)
    ]
    pricing = data.get("pricing") if isinstance(data.get("pricing"), dict) else {}
    offers = data.get("offers") if isinstance(data.get("offers"), dict) else {}
    restaurant = data.get("restaurant") if isinstance(data.get("restaurant"), dict) else None

    result = {
        "status": "success" if items else "empty",
        "cart_id": data.get("cart_id"),
        "restaurant": restaurant,
        "items": items,
        "item_count": data.get("item_count", len(items)),
        "pricing": pricing,
        # coupon_discount==0 means "auto-suggested", not actually applied (per docs).
        "coupon_applied": offers.get("coupon_applied") if (offers.get("coupon_discount") or 0) > 0 else None,
        "coupon_discount": offers.get("coupon_discount", 0),
        "address_id": raw.get("addressId", address_id),
        "message": None,
    }
    debug("food_mcp", f"get_food_cart parsed (json mode): status={result['status']!r} item_count={len(items)}")
    return result


def update_cart(
    restaurant_id: str, address_id: str, cart_items: list[dict[str, Any]], restaurant_name: str | None = None
) -> dict[str, Any]:
    """update_food_cart -- add/change items. Docs: renders no cart view
    itself ("does NOT render any widget or cart UI"), so the documented flow
    is update_food_cart -> get_food_cart; callers here should follow with
    `get_cart` to show the result and confirm the add actually took.

    `cart_items` entries need {"menu_item_id": <id>, "quantity": <int>} --
    NOT bare "id" (an earlier version of this used "id"; a real call with
    it returned 'Cart updated.\nCart is empty.', i.e. accepted but silently
    a no-op, because Swiggy didn't recognise the field. Confirmed via
    Swiggy's own docs.md source, which explicitly ties menu_item_id to this
    tool: '"menu_item_id": Food menu item identifier used by
    update_food_cart.'). Items with mandatory variants/add-ons ("Items use
    either variants or variantsV2 format") may still need one of those two
    fields too -- not yet confirmed, since Cravr doesn't capture per-item
    variant/addon options today -- so both the request and response are
    always logged to catch that from a real run.
    """
    arguments: dict[str, Any] = {
        "restaurantId": restaurant_id,
        "addressId": address_id,
        "cartItems": cart_items,
    }
    if restaurant_name:
        arguments["restaurantName"] = restaurant_name
    debug("food_mcp", f"update_food_cart request: {arguments!r}")

    try:
        raw = _unwrap(call_tool_sync("update_food_cart", arguments))
    except Exception as exc:
        debug("food_mcp", f"update_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to update cart: {exc}"}

    debug("food_mcp", f"raw update_food_cart response: {raw!r}")

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        debug("food_mcp", f"update_food_cart parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"update_food_cart parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    result = {"status": "success", "message": raw.get("message") if isinstance(raw, dict) else None}
    debug("food_mcp", f"update_food_cart parsed (json mode): {result!r}")
    return result


def apply_coupon(coupon_code: str, address_id: str, cart_id: str | None = None) -> dict[str, Any]:
    """apply_food_coupon -> {"status", "message"}. Caller should re-fetch
    via `get_cart` afterwards to see the updated pricing/coupon_discount."""
    arguments: dict[str, Any] = {"couponCode": coupon_code, "addressId": address_id}
    if cart_id:
        arguments["cartId"] = cart_id
    debug("food_mcp", f"apply_food_coupon request: {arguments!r}")

    try:
        raw = _unwrap(call_tool_sync("apply_food_coupon", arguments))
    except Exception as exc:
        debug("food_mcp", f"apply_food_coupon FAILED: {exc}")
        return {"status": "error", "message": f"Unable to apply coupon: {exc}"}

    debug("food_mcp", f"raw apply_food_coupon response: {raw!r}")

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        debug("food_mcp", f"apply_food_coupon parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"apply_food_coupon parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    result = {"status": "success", "message": raw.get("message") if isinstance(raw, dict) else None}
    debug("food_mcp", f"apply_food_coupon parsed (json mode): {result!r}")
    return result


def fetch_coupons(restaurant_id: str, address_id: str, coupon_code: str | None = None) -> dict[str, Any]:
    """fetch_food_coupons -> {"status", "coupons", "count", "message"}. Docs:
    only COD-applicable coupons are returned. Each coupon: {"id", "title",
    "subtitle", "description", "applicable", "applicability_status", "terms"}
    -- populated only on the documented-JSON path; if the real tool turns
    out to be plain text too (unconfirmed as of this writing, unlike the
    other four cart tools which are all now known to be text), `message`
    carries the raw status and `coupons` stays empty rather than crashing.
    """
    arguments: dict[str, Any] = {"restaurantId": restaurant_id, "addressId": address_id}
    if coupon_code:
        arguments["couponCode"] = coupon_code
    debug("food_mcp", f"fetch_food_coupons request: {arguments!r}")

    try:
        raw = _unwrap(call_tool_sync("fetch_food_coupons", arguments))
    except Exception as exc:
        debug("food_mcp", f"fetch_food_coupons FAILED: {exc}")
        return {"status": "error", "message": f"Unable to fetch coupons: {exc}", "coupons": []}

    debug("food_mcp", f"raw fetch_food_coupons response: {raw!r}")

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        result = {"status": "text", "message": message, "coupons": []}
        debug("food_mcp", f"fetch_food_coupons parsed (text mode): {result!r}")
        return result

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"fetch_food_coupons parsed: status='error' message={error!r}")
        return {"status": "error", "message": error, "coupons": []}
    if not isinstance(raw, dict):
        return {"status": "error", "message": "Unexpected fetch_food_coupons response", "coupons": []}

    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    coupons: list[dict[str, Any]] = []
    for section in data.get("coupon_sections") or []:
        if not isinstance(section, dict):
            continue
        for c in section.get("coupons") or []:
            if not isinstance(c, dict):
                continue
            coupons.append({
                "id": c.get("id"),
                "title": c.get("title"),
                "subtitle": c.get("subtitle"),
                "description": c.get("description"),
                "applicable": c.get("applicable", c.get("applicabilityStatus") == "APPLICABLE"),
                "applicability_status": c.get("applicabilityStatus"),
                "terms": c.get("terms"),
            })

    if not coupons:
        result = {"status": "no_results", "message": data.get("status_message") or "No coupons available", "coupons": []}
        debug("food_mcp", f"fetch_food_coupons parsed (json mode): {result!r}")
        return result
    result = {"status": "success", "coupons": coupons, "count": len(coupons)}
    debug("food_mcp", f"fetch_food_coupons parsed (json mode): {len(coupons)} coupon(s)")
    return result


def flush_cart() -> dict[str, Any]:
    """flush_food_cart -- clears the entire cart. No parameters (auth is
    session-scoped, per docs)."""
    debug("food_mcp", "flush_food_cart request: {}")
    try:
        raw = _unwrap(call_tool_sync("flush_food_cart", {}))
    except Exception as exc:
        debug("food_mcp", f"flush_food_cart FAILED: {exc}")
        return {"status": "error", "message": f"Unable to clear cart: {exc}"}

    debug("food_mcp", f"raw flush_food_cart response: {raw!r}")

    if isinstance(raw, str):
        message = _strip_widget_disclaimer(raw)
        debug("food_mcp", f"flush_food_cart parsed (text mode): message={message!r}")
        return {"status": "success", "message": message}

    error = _check_success(raw)
    if error:
        debug("food_mcp", f"flush_food_cart parsed: status='error' message={error!r}")
        return {"status": "error", "message": error}
    debug("food_mcp", "flush_food_cart parsed (json mode): success")
    return {"status": "success"}
