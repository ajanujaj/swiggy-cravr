from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..cart.CartAgent import add_dish_to_cart, _current_restaurant_id, _current_restaurant_name
from ..cart.guards import _menu_categories
from ..client import call_tool_sync, using_real_food_mcp
from ..debug_utils import debug
from ..intent import KNOWN_CUISINE_WORDS, NON_VEG_ANSWERS, VEG_ANSWERS, extract_tastes
from ..intent.meal_utils import CRAVING_TAG_CUES
from ..profile_utils import latest_user_text
from ..state import CravrState

DIETARY_TYPE_BY_PREFERENCE = {
    "veg": "Veg",
    "vegetarian": "Veg",
    "pure veg": "Veg",
    "veggie": "Veg",
    "vegan": "Veg",
    "non-veg": "Non-Veg",
    "non veg": "Non-Veg",
    "nonveg": "Non-Veg",
    "non-vegetarian": "Non-Veg",
    "non vegetarian": "Non-Veg",
    "eggetarian": "Non-Veg",
    "meat": "Non-Veg",
}

KNOWN_TAGS = {
    "breakfast", "comfort-food", "dessert", "gluten-free", "high-fiber",
    "high-protein", "lightweight", "low-calorie", "low-carb", "mild",
    "one-pot-meal", "organic", "savory", "spicy", "sugar-free-option",
    "sweet", "tangy", "vegan", "vegan-option", "vegetarian",
}

BEVERAGE_CUES = ("coffee", "tea", "chai", "drink", "beverage", "juice", "smoothie",
                 "latte", "cappuccino", "espresso", "shake", "lassi")
DESSERT_CUES = ("dessert", "sweet dish", "pudding", "cake", "brownie", "ice cream",
                "sweet tooth", "something sweet", "pastry")
SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "small plate",
             "nibble", "finger food")

MEAL_COURSES = "main,side"

DISPLAY_COUNT = 3

ACCEPTANCE_TOKENS = ("yes", "yep", "yeah", "sure", "ok", "okay", "perfect",
                     "great", "sounds good", "looks good", "love it",
                     "go ahead", "that works", "book it", "accept")
REJECTION_TOKENS = ("no", "nope", "nah", "not that", "different", "another",
                    "change", "new one", "something else", "skip", "reject")
MORE_TOKENS = ("more", "other", "others", "else", "next", "alternatives",
               "alternative", "rest", "again")


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    return latest_user_text(messages)


def _mentions(user_input: str, tokens: tuple[str, ...]) -> bool:
    normalized = (user_input or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized) for token in tokens)


def _looks_like_acceptance(user_input: str) -> bool:
    return _mentions(user_input, ACCEPTANCE_TOKENS) and not _mentions(user_input, REJECTION_TOKENS)


def _looks_like_rejection(user_input: str) -> bool:
    return _mentions(user_input, REJECTION_TOKENS)


def _looks_like_more_request(user_input: str) -> bool:
    return _mentions(user_input, MORE_TOKENS)


def _describe(dish: dict[str, Any]) -> str:
    details = [str(dish.get("cuisine") or "")]
    if dish.get("price_inr") is not None:
        details.append(f"₹{dish['price_inr']}")
    if dish.get("rating") is not None:
        details.append(f"rating {dish['rating']}")
    detail_text = ", ".join(d for d in details if d)
    return (
        f"{dish.get('item_name') or 'a dish'} from {dish.get('restaurant_name') or 'the restaurant'}"
        f"{f' ({detail_text})' if detail_text else ''}"
    )


def _build_recommendation_text(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return (
            "I could not find a matching food suggestion right now. "
            "I can try a different cuisine or budget if you want."
        )

    if len(recommendations) == 1:
        return (
            f"I’d suggest {_describe(recommendations[0])} — that's the only match "
            f"for what you asked.\n{ONLY_MATCH_PROMPT}"
        )

    return (
        f"Here are {len(recommendations)} options:\n{_numbered(recommendations)}\n{PICK_PROMPT}"
    )


PICK_PROMPT = "Pick one by number or name, ask for more, or tell me what to change."

ONLY_MATCH_PROMPT = "    1. Yes, I'll take it\n    2. Widen the search"


def _numbered(dishes: list[dict[str, Any]]) -> str:
    return "\n".join(f"  {i}. {_describe(d)}" for i, d in enumerate(dishes, start=1))


def _dish_pick_options(dishes: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"value": str(i), "label": d.get("item_name") or f"Option {i}"}
        for i, d in enumerate(dishes, start=1)
    ]


ONLY_MATCH_OPTIONS = [
    {"value": "1", "label": "Yes, I'll take it"},
    {"value": "2", "label": "Widen the search"},
]


ORDINAL_WORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}
CARDINAL_WORDS = {"one": 1, "two": 2, "three": 3}


def _resolve_choice(user_input: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = (user_input or "").strip().lower()
    if not normalized or not options:
        return None

    for word, position in ORDINAL_WORDS.items():
        if position <= len(options) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized):
            return options[position - 1]

    for word, position in CARDINAL_WORDS.items():
        if position > len(options):
            continue
        if normalized.strip(" .!") == word or re.search(
            rf"(?:number|option|#)\s*{re.escape(word)}(?!\w)", normalized
        ):
            return options[position - 1]

    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", normalized)
    if match and 1 <= int(match.group(1)) <= len(options):
        return options[int(match.group(1)) - 1]

    # Exact match against the whole name, whatever its length -- catches
    # short names (a menu category like "Rice" or "Naan", 4 letters) that
    # the >=5-letter word filter below would otherwise miss entirely.
    for dish in options:
        if str(dish.get("item_name") or "").strip().lower() == normalized:
            return dish

    for dish in options:
        words = [w for w in re.findall(r"\w+", str(dish.get("item_name") or "").lower()) if len(w) >= 5]
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", normalized) for w in words):
            return dish

    return None


# Sent verbatim by the Chainlit UI when the user submits a multi-select
# offer (OptionChips.jsx's `multi` mode) with nothing checked -- the
# "Skip -- none of these" button. Mirrors UI/components.py's
# _NONE_ADDON_VALUE / cart/add_flow.py's _NONE_ADDON_VALUE (same literal,
# kept as its own constant here since RecommendAgent doesn't otherwise
# depend on either of those modules).
_NO_SELECTION_VALUE = "__no_addon__"


def _resolve_multiple_choices(
    user_input: str, options: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Like _resolve_choice, but for a multi-select submission -- a
    comma-joined string of chip values (see UI/components.py's
    multiselect_answer, fired by OptionChips.jsx's multi-select submit
    button) or free-typed text naming several options ("Breads and Rice",
    "Breads, Rice"). Splits on comma/"and"/"&", resolves each token with
    the same matching rules as a single _resolve_choice pick, preserves
    the order picked, and drops duplicates. A plain single-token input (an
    ordinary one-at-a-time reply) still resolves exactly like
    _resolve_choice, so this is a safe drop-in wherever a category offer
    might get either kind of reply."""
    if not (user_input or "").strip() or not options:
        return []

    tokens = [t.strip() for t in re.split(r",|\band\b|&", user_input, flags=re.IGNORECASE) if t.strip()]
    chosen: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for token in tokens:
        match = _resolve_choice(token, options)
        if match is not None and id(match) not in seen_ids:
            seen_ids.add(id(match))
            chosen.append(match)
    return chosen


def _course_preference(intent_profile: dict[str, Any]) -> str:
    haystack = " ".join(
        str(value).lower()
        for value in (
            intent_profile.get("craving"),
            intent_profile.get("cuisine"),
            intent_profile.get("meal_time"),
            *(intent_profile.get("other_constraints") or []),
        )
        if value
    )

    if any(cue in haystack for cue in BEVERAGE_CUES):
        return "beverage"
    if any(cue in haystack for cue in DESSERT_CUES):
        return "dessert"
    if "sweet" in [str(t).lower() for t in intent_profile.get("taste_preference") or []]:
        return "dessert,main"
    if any(cue in haystack for cue in SIDE_CUES):
        return "side"
    return "main"


def _build_recommendation_filters(intent_profile: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    tag_terms: list[str] = []
    for term in [intent_profile.get("craving"), intent_profile.get("craving_tag"), *(intent_profile.get("taste_preference") or [])]:
        if not term:
            continue
        normalized = str(term).strip().lower()
        if normalized in KNOWN_TAGS and normalized not in tag_terms:
            tag_terms.append(normalized)
    if tag_terms:
        filters["tags"] = ",".join(tag_terms)

    if intent_profile.get("craving"):
        craving = str(intent_profile["craving"])
        filters["craving"] = craving
        craving_tag = intent_profile.get("craving_tag")
        # craving_tag ("what round 1 said") can already BE a specific,
        # multi-word dish name (e.g. "Palak Paneer"), with round 2's pick a
        # pairing/side (e.g. "Naan"), not a narrower version of it -- using
        # craving alone then searches for the side only, silently dropping
        # the dish actually asked for (live regression, 2026-09-28: "Palak
        # Paneer" -> "Naan" searched for "Naan" only, "giving only Naan").
        # A single combined search string doesn't fix this either -- tried
        # live and confirmed worse: "Palak Paneer Naan" ranked "Paneer
        # Naan" (a different, unrelated dish that just happens to share two
        # of the three words) above genuine Palak Paneer options, because
        # Swiggy's free-text search doesn't reliably weigh multi-concept
        # queries. So both are searched separately instead (see
        # search.py's craving_secondary handling) and ranking picks the
        # better match -- see _rank_candidates' craving-relevance term.
        #
        # Only promote craving_tag to a secondary search when it reads as
        # an already-specific dish: 2+ words rules out a single vague
        # category word like "sandwich" (round 2 narrowing that down to
        # "Grilled Cheese" must still win outright, not get diluted by
        # re-admitting every other kind of sandwich), and it must not be a
        # cuisine name (that's a restaurant-level filter, not a dish, and
        # search_menu returns poor results for a bare cuisine query -- see
        # the "North Indian" -> 272-unfiltered-options regression) or one
        # of CRAVING_TAG_CUES' generic mood tags (nothing to search for on
        # its own, e.g. "comfort-food") or already the same dish as craving.
        if (
            craving_tag
            and len(str(craving_tag).split()) >= 2
            and str(craving_tag) not in CRAVING_TAG_CUES
            and str(craving_tag).strip().lower() not in KNOWN_CUISINE_WORDS
            and str(craving_tag).strip().lower() != craving.strip().lower()
        ):
            filters["craving_secondary"] = str(craving_tag)

    cuisine = intent_profile.get("cuisine")
    if cuisine:
        filters["cuisine"] = str(cuisine)

    filters["course"] = _course_preference(intent_profile)

    for preference in intent_profile.get("dietary_preferences") or []:
        dietary_type = DIETARY_TYPE_BY_PREFERENCE.get(str(preference).strip().lower())
        if dietary_type:
            filters["dietary_type"] = dietary_type
            break

    budget = intent_profile.get("budget_inr")
    if budget is not None:
        try:
            filters["max_price_inr"] = int(budget)
        except (TypeError, ValueError):
            pass

    return filters


def _summarise_history(history_profile: dict[str, Any]) -> dict[str, Any]:
    orders = (history_profile or {}).get("orders") or []
    cuisines = Counter(
        str(o.get("cuisine")).strip() for o in orders if o.get("cuisine")
    )
    items = Counter(str(o.get("item_name")).strip() for o in orders if o.get("item_name"))

    return {
        "order_count": len(orders),
        "top_cuisines": [name for name, _count in cuisines.most_common(2)],
        "repeat_items": [name for name, count in items.most_common(3) if count > 1],
    }


_DIETARY_WORD = {"Veg": "vegetarian", "Non-Veg": "non-vegetarian"}


def _brief_summary(filters: dict[str, Any], history: dict[str, Any]) -> str:
    course = filters.get("course", "main").split(",")[0]
    descriptors = [
        *(t for t in str(filters.get("tags") or "").split(",") if t),
        _DIETARY_WORD.get(filters.get("dietary_type", ""), ""),
        str(filters.get("cuisine") or ""),
    ]
    phrase = " ".join(d for d in descriptors if d)
    intent = f"{phrase} {course}".strip() if phrase else f"any {course}"
    if filters.get("max_price_inr") is not None:
        intent += f" under ₹{filters['max_price_inr']}"

    parts = [intent]
    if history.get("top_cuisines"):
        parts.append("usually orders " + " and ".join(history["top_cuisines"]))
    if history.get("repeat_items"):
        parts.append("re-orders " + ", ".join(history["repeat_items"]))

    return "; ".join(parts) + "."


def _apply_saved_preferences(
    intent_profile: dict[str, Any], preference_profile: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(intent_profile)
    preferences = _coerce_profile(preference_profile).get("preferences") or {}
    if isinstance(preferences, list):
        preferences = preferences[0] if preferences else {}
    if not isinstance(preferences, dict):
        return merged

    if merged.get("budget_inr") is None and preferences.get("default_budget_inr") is not None:
        merged["budget_inr"] = preferences["default_budget_inr"]
    if not merged.get("dietary_preferences") and preferences.get("dietary_preference"):
        merged["dietary_preferences"] = [preferences["dietary_preference"]]
    return merged


def build_recommendation_brief(
    intent_profile: dict[str, Any],
    history_profile: dict[str, Any],
    preference_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_intent = _apply_saved_preferences(
        _coerce_profile(intent_profile), preference_profile
    )
    filters = _build_recommendation_filters(effective_intent)
    history = _summarise_history(_coerce_profile(history_profile))
    return {"filters": filters, "history": history, "summary": _brief_summary(filters, history)}


def _filter_variants(filters: dict[str, Any]) -> list[dict[str, Any]]:
    course_steps = [filters.get("course")]
    if filters.get("course") == "main":
        course_steps.append(MEAL_COURSES)

    variants: list[dict[str, Any]] = []
    for dropped in ((), ("cuisine",), ("tags",), ("cuisine", "tags")):
        for course in course_steps:
            variant = {k: v for k, v in filters.items() if k not in dropped}
            if course:
                variant["course"] = course
            if variant not in variants:
                variants.append(variant)

    return variants


def _craving_match_score(item_name: str, *phrases: str | None) -> int:
    """+2 for each given phrase the item's name matches exactly, +1 for
    each it merely contains as a substring -- across every phrase passed,
    not just one. Used to rank a merged pool of results pulled from two
    separate searches (craving + craving_secondary -- see
    _build_recommendation_filters) so a dish genuinely named after what was
    asked for outranks one that only shares a single stray word with one
    of the two phrases (live regression, 2026-09-28: without this, a
    same-tier boolean relevance check let "Paneer Naan" -- matches only
    "Naan" -- rank ahead of "Palak Paneer" -- matches "Palak Paneer"
    exactly -- since both merely "passed" the relevance filter with no
    notion of how well)."""
    name = (item_name or "").strip().lower()
    score = 0
    for phrase in phrases:
        phrase_l = str(phrase or "").strip().lower()
        if not phrase_l:
            continue
        if name == phrase_l:
            score += 2
        elif phrase_l in name:
            score += 1
    return score


def _rank_candidates(
    recommendations: list[dict[str, Any]],
    desired_tags: list[str],
    desired_cuisine: str | None,
    desired_course: str | None,
    history: dict[str, Any],
    desired_craving: str | None = None,
    desired_craving_secondary: str | None = None,
) -> list[dict[str, Any]]:
    wanted_tags = {t.strip().lower() for t in desired_tags if t}
    cuisine = (desired_cuisine or "").strip().lower()
    preferred_course = (desired_course or "").split(",")[0].strip().lower()

    past_cuisines = {str(c).strip().lower() for c in (history or {}).get("top_cuisines") or []}

    def sort_key(item: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
        item_tags = {t.strip().lower() for t in str(item.get("tags") or "").split(";") if t.strip()}
        item_cuisine = str(item.get("cuisine") or "").strip().lower()
        item_course = str(item.get("course") or "").strip().lower()
        try:
            rating = float(item.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0.0
        return (
            _craving_match_score(
                str(item.get("item_name") or ""), desired_craving, desired_craving_secondary
            ),
            1 if preferred_course and item_course == preferred_course else 0,
            len(wanted_tags & item_tags),
            1 if cuisine and cuisine in item_cuisine else 0,
            1 if item_cuisine in past_cuisines else 0,
            rating,
        )

    return sorted(recommendations, key=sort_key, reverse=True)

def _unwrap_tool_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("recommendations") or []
        debug("recommendation_agent", f"tool returned status={result.get('status')!r}")
        return []

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            item_text = getattr(item, "text", None)
            if not isinstance(item_text, str):
                continue
            try:
                payload = json.loads(item_text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                if payload.get("status") == "success":
                    return payload.get("recommendations") or []
                debug("recommendation_agent", f"tool returned status={payload.get('status')!r}")

    return []


def _resolve_address_id(address_id: str | None) -> str | None:
    if address_id:
        return address_id
    return food_mcp.get_saved_address().get("address_id")


def _call_recommendation_tool(filters: dict[str, Any], address_id: str | None = None) -> list[dict[str, Any]]:
    if using_real_food_mcp():
        return food_mcp.search_dishes(filters, address_id=_resolve_address_id(address_id))

    mock_filters = {k: v for k, v in filters.items() if k != "craving"}
    try:
        result = call_tool_sync("recommendation", mock_filters)
    except Exception as exc:
        debug("recommendation_agent", f"MCP call FAILED ({type(exc).__name__}: {exc})")
        return []

    return _unwrap_tool_result(result)


def fetch_recommendations(
    brief: dict[str, Any], address_id: str | None = None, exclude_restaurant_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    brief = _coerce_profile(brief)
    filters = dict(brief.get("filters") or {})
    history = _coerce_profile(brief.get("history"))
    exclude_ids = {str(r) for r in (exclude_restaurant_ids or ())}
    debug("recommendation_agent", f"brief summary={brief.get('summary')!r}")
    debug("recommendation_agent", f"strict filters={filters} history={history}")

    desired_tags = [t for t in str(filters.get("tags") or "").split(",") if t]
    desired_cuisine = filters.get("cuisine")
    desired_course = filters.get("course")

    for attempt, variant in enumerate(_filter_variants(filters), start=1):
        recommendations = _call_recommendation_tool(variant, address_id=address_id)
        if exclude_ids:
            recommendations = [
                r for r in recommendations if str(r.get("restaurant_id")) not in exclude_ids
            ]
        debug("recommendation_agent", f"attempt {attempt} filters={variant} -> {len(recommendations)} result(s)")
        if recommendations:
            ranked = _rank_candidates(
                recommendations, desired_tags, desired_cuisine, desired_course, history,
                desired_craving=filters.get("craving"),
                desired_craving_secondary=filters.get("craving_secondary"),
            )
            relaxed = sorted(set(filters) - set(variant))
            if variant.get("course") != desired_course:
                relaxed.append(f"course->{variant.get('course')}")
            debug(
                "recommendation_agent",
                f"top pick={ranked[0].get('item_name')!r} ({ranked[0].get('course')}) "
                f"(relaxed: {relaxed or 'nothing'}, ranked against tags={desired_tags} "
                f"cuisine={desired_cuisine!r} course={desired_course!r})",
            )
            return ranked

    debug("recommendation_agent", "no recommendations after exhausting all filter variants")
    return []


def widen_recommendations(
    brief: dict[str, Any], address_id: str | None = None, exclude_items: set[str] | None = None
) -> list[dict[str, Any]]:
    brief = _coerce_profile(brief)
    filters = dict(brief.get("filters") or {})
    history = _coerce_profile(brief.get("history"))
    exclude = {str(name).strip().lower() for name in (exclude_items or set())}

    desired_tags = [t for t in str(filters.get("tags") or "").split(",") if t]
    desired_cuisine = filters.get("cuisine")
    desired_course = filters.get("course")

    for attempt, variant in enumerate(_filter_variants(filters)[1:], start=1):
        recommendations = _call_recommendation_tool(variant, address_id=address_id)
        recommendations = [
            item for item in recommendations
            if str(item.get("item_name") or "").strip().lower() not in exclude
        ]
        debug("recommendation_agent", f"widen attempt {attempt} filters={variant} -> {len(recommendations)} result(s)")
        if recommendations:
            return _rank_candidates(
                recommendations, desired_tags, desired_cuisine, desired_course, history,
                desired_craving=filters.get("craving"),
                desired_craving_secondary=filters.get("craving_secondary"),
            )

    debug("recommendation_agent", "widen found nothing beyond the original match")
    return []


def _offer_category_batch(
    lead_in: str, categories: list[dict[str, Any]], restaurant_scope: dict[str, Any] | None = None,
) -> dict:
    """Offer a restaurant's menu categories (Breads, Rice, Chinese, ...) as
    multi-select pick options -- the user can check several at once (see
    UI/graph_bridge.py routing offer_kind == "category" to
    components.send_multiselect_message) before the browsing step in
    _offer_dish_batch shows the combined dishes from every category picked
    (see _handle_category_pick). `categories` is _menu_categories()' shape:
    [{"category": str, "items": [...]}, ...]. Shows every category at once
    (not capped to DISPLAY_COUNT) since the whole point is to let the user
    see the restaurant's breadth in one glance."""
    summaries = [{"item_name": c["category"], "category": c["category"]} for c in categories]
    lines = "\n".join(
        f"  {i}. {c['category']} ({len(c['items'])} item{'' if len(c['items']) == 1 else 's'})"
        for i, c in enumerate(categories, start=1)
    )
    # Rendered as a multi-select, which only shows this content's FIRST
    # LINE (the chips themselves list every category) -- kept in full here
    # anyway since debug logs, and any non-chip fallback rendering, still
    # read the whole thing.
    content = f"{lead_in}\n{lines}\nPick as many as you like, then hit submit." if lead_in else lines
    return _reply(
        content,
        awaiting_feedback=True, satisfied=False,
        last_recommendations=summaries, alternates=[], filters={},
        categories=[c["items"] for c in categories],
        offer_kind="category",
        options=_dish_pick_options(summaries),
        restaurant_scope=restaurant_scope,
    )


def _offer_dish_batch(
    lead_in: str, items: list[dict[str, Any]], restaurant_scope: dict[str, Any] | None = None,
    filters: dict[str, Any] | None = None,
) -> dict:
    """Offer a restaurant-scoped batch of dishes (e.g. everything under one
    menu category) the same way a fresh catalog search would, but carrying
    restaurant_scope forward so a decline widens within this restaurant --
    see _widen_within_restaurant -- rather than falling back to a stale,
    unrelated catalog-wide search."""
    batch, rest = items[:DISPLAY_COUNT], items[DISPLAY_COUNT:]
    if len(batch) == 1:
        content = _build_recommendation_text(batch)
    elif lead_in:
        content = f"{lead_in}\n{_numbered(batch)}\n{PICK_PROMPT}"
    else:
        content = _build_recommendation_text(batch)
    return _reply(
        content,
        awaiting_feedback=True, satisfied=False,
        last_recommendations=batch, alternates=rest, filters=filters or {},
        options=ONLY_MATCH_OPTIONS if len(batch) == 1 else _dish_pick_options(batch),
        restaurant_scope=restaurant_scope,
    )


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _handle_category_pick(
    chosen: list[dict[str, Any]], recommendation_state: dict[str, Any], address_id: str | None,
) -> dict:
    """The user picked one or more of the categories _offer_category_batch
    offered (OptionChips' multi-select lets them check several at once) --
    merge the picked categories' items into one combined pick-list, instead
    of treating any category itself as something to add to the cart."""
    offered = recommendation_state.get("last_recommendations") or []
    categories = recommendation_state.get("categories") or []
    restaurant_scope = dict(recommendation_state.get("restaurant_scope") or {})

    labels: list[str] = []
    seen_ids: set[str] = set()
    items: list[dict[str, Any]] = []
    for pick in chosen:
        try:
            index = offered.index(pick)
        except ValueError:
            continue
        label = pick.get("category") or pick.get("item_name") or "that category"
        labels.append(label)
        for item in categories[index] if 0 <= index < len(categories) else []:
            item_id = str(item.get("menu_item_id"))
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            items.append(item)

    if not items:
        label_text = _join_labels(labels) if labels else "that category"
        return _reply(
            f"Couldn't find items under {label_text} right now -- try again in a moment.",
            awaiting_feedback=False, satisfied=False,
        )

    restaurant_scope["current_category"] = labels
    return _offer_dish_batch(f"Here's what's under {_join_labels(labels)}:", items, restaurant_scope)


def _widen_within_restaurant(
    restaurant_scope: dict[str, Any], offered: list[dict[str, Any]], address_id: str | None,
) -> dict | None:
    """A restaurant-scoped 'only match, widen' fallback -- returns a full
    reply dict, or None if there's genuinely nothing else to show.

    Used when the single offer being declined came from a restaurant-scoped
    context (an "add more from this restaurant" category browse, or a
    single dish inside one of its categories -- see cart/flows.py's
    _suggest_more_from_restaurant and _handle_category_pick above), not a
    fresh catalog-wide search. Falling through to the generic
    widen_recommendations() there used the stale, unrelated
    recommendation_brief left over from however the user originally found
    this restaurant -- searching the whole Swiggy catalog again and
    surfacing a same-named dish from a completely different restaurant
    (live regression, 2026-09-29: "add more items" after adding Palak
    Paneer from Punjabi Angithi widened into "Palak Paneer from Punjab
    Grill" -- a different restaurant entirely).

    Re-shows the restaurant's OTHER menu categories (excluding whichever
    one was just declined, and anything already in the cart) rather than
    guessing one more complementary dish -- that's how the user actually
    wants to browse a restaurant (see _offer_category_batch).
    """
    restaurant_id = restaurant_scope.get("restaurant_id")
    if not restaurant_id:
        return None

    menu_items = food_mcp.get_restaurant_menu_items(
        restaurant_id, address_id, restaurant_scope.get("restaurant_name") or ""
    )
    if not menu_items:
        return None

    exclude_ids = {str(i) for i in (restaurant_scope.get("existing_ids") or [])}
    exclude_ids |= {str(d.get("menu_item_id")) for d in offered if d.get("menu_item_id")}
    categories = _menu_categories(menu_items, exclude_ids)

    current_categories = restaurant_scope.get("current_category")
    if current_categories:
        # Stamped by _handle_category_pick as a list (one or more categories
        # picked in the same multi-select submission); tolerate a plain str
        # too, in case an older single-category restaurant_scope is still
        # floating around in a resumed session's checkpoint.
        if isinstance(current_categories, str):
            current_categories = [current_categories]
        excluded = {str(c) for c in current_categories}
        categories = [c for c in categories if c["category"] not in excluded]
    if not categories:
        return None

    next_scope = {k: v for k, v in restaurant_scope.items() if k != "current_category"}
    return _offer_category_batch("Here's what else this restaurant has:", categories, next_scope)


def _carries_new_constraints(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower()
    if re.search(r"\d{2,}", normalized):
        return True
    if extract_tastes(normalized):
        return True
    if any(word in normalized for word in KNOWN_CUISINE_WORDS):
        return True
    return _mentions(normalized, VEG_ANSWERS + NON_VEG_ANSWERS)


WIDEN_TOKENS = ("widen", "broaden", "expand", "wider")


def _looks_like_decline_only_offer(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower().rstrip(".!")
    if normalized == "2":
        return True
    return _looks_like_rejection(user_input) or _mentions(user_input, WIDEN_TOKENS)


def is_pure_feedback(user_input: str, recommendation_state: dict[str, Any] | None) -> bool:
    state = recommendation_state or {}
    offered = state.get("last_recommendations") or []
    if not offered or not state.get("awaiting_feedback") or state.get("satisfied"):
        return False

    # The multi-select "Skip -- none of these" submission never resolves to
    # a pick or matches any of the acceptance/rejection wording below (its
    # literal text doesn't read like either), but it's still definitely
    # feedback on the current offer, not a fresh request -- see
    # recommendation_agent_node's is_category_offer branch.
    if (user_input or "").strip() == _NO_SELECTION_VALUE:
        return True

    if _resolve_choice(user_input, offered):
        return True
    if len(offered) == 1 and (user_input or "").strip().rstrip(".!") == "2":
        return True
    if _carries_new_constraints(user_input):
        return False
    return (
        _looks_like_acceptance(user_input)
        or _looks_like_rejection(user_input)
        or _looks_like_more_request(user_input)
    )


def _reply(content: str, **rec_state: Any) -> dict:
    return {
        "recommendation_state": rec_state,
        "messages": [AIMessage(content=content, name="recommendation_agent")],
    }


def _reply_with_pick(
    chosen: dict[str, Any], address_id: str | None, state: CravrState | None = None, **rec_state: Any
) -> dict:
    lead_in = f"Great — {_describe(chosen)} it is."
    if not address_id:
        return _reply(
            f"{lead_in} I don't have a delivery address yet, so I couldn't add it to your cart.",
            **rec_state,
        )
    known_restaurant_id = _current_restaurant_id(state) if state is not None else None
    known_restaurant_name = _current_restaurant_name(state) if state is not None else None
    result = add_dish_to_cart(
        chosen, address_id, lead_in=lead_in,
        known_restaurant_id=known_restaurant_id, known_restaurant_name=known_restaurant_name,
    )
    result["recommendation_state"] = rec_state
    return result


def _serve_next_batch(
    lead_in: str,
    offered: list[dict[str, Any]],
    alternates: list[dict[str, Any]],
    filters: dict[str, Any],
    restaurant_scope: dict[str, Any] | None = None,
) -> dict:
    if alternates:
        batch, rest = alternates[:DISPLAY_COUNT], alternates[DISPLAY_COUNT:]
        debug("recommendation_agent", f"serving {len(batch)} more, {len(rest)} still held back")
        return _reply(
            f"{lead_in}\n{_numbered(batch)}\n{PICK_PROMPT}",
            awaiting_feedback=True, satisfied=False,
            last_recommendations=batch, alternates=rest, filters=filters,
            options=_dish_pick_options(batch),
            restaurant_scope=restaurant_scope,
        )

    debug("recommendation_agent", "no alternates left to offer")
    return _reply(
        "That's everything on the menu matching what you asked for. I can widen it — "
        "a different cuisine, a higher budget, or including sides and desserts?",
        awaiting_feedback=True, satisfied=False,
        last_recommendations=offered, alternates=[], filters=filters,
        restaurant_scope=restaurant_scope,
    )


def recommendation_agent_node(state: CravrState) -> dict:
    brief = _coerce_profile(state.get("recommendation_brief"))
    if not brief:
        debug("recommendation_agent", "no brief in state — deriving one from the profiles")
        brief = build_recommendation_brief(
            _coerce_profile(state.get("intent_profile")),
            _coerce_profile(state.get("history_profile")),
        )

    brief_filters = dict(brief.get("filters") or {})
    address_id = _coerce_profile(state.get("context_profile")).get("address_id")
    recommendation_state = dict(state.get("recommendation_state") or {})
    awaiting_feedback = bool(recommendation_state.get("awaiting_feedback"))
    offered = recommendation_state.get("last_recommendations") or []
    alternates = recommendation_state.get("alternates") or []
    satisfied = bool(recommendation_state.get("satisfied"))

    user_input = _extract_latest_user_input(state.get("messages") or [])
    debug("recommendation_agent", f"user_input={user_input!r} awaiting_feedback={awaiting_feedback}")

    if awaiting_feedback and not satisfied:
        is_category_offer = recommendation_state.get("offer_kind") == "category"
        if is_category_offer:
            if user_input.strip() == _NO_SELECTION_VALUE:
                debug("recommendation_agent", "user skipped every category option")
                restaurant_scope = recommendation_state.get("restaurant_scope")
                if restaurant_scope and restaurant_scope.get("restaurant_id"):
                    return _reply(
                        "No problem -- let me know if you'd like to see something else from "
                        "this restaurant, or somewhere else entirely.",
                        awaiting_feedback=False, satisfied=False, last_recommendations=offered,
                    )
                return _reply(
                    "No problem \u2014 tell me what you want to change, such as cuisine, budget, "
                    "or dietary preference.",
                    awaiting_feedback=False, satisfied=False, last_recommendations=offered,
                )
            chosen_many = _resolve_multiple_choices(user_input, offered)
            if chosen_many:
                debug(
                    "recommendation_agent",
                    f"user picked {len(chosen_many)} categor"
                    f"{'y' if len(chosen_many) == 1 else 'ies'}: "
                    f"{[c.get('category') for c in chosen_many]}",
                )
                return _handle_category_pick(chosen_many, recommendation_state, address_id)
        else:
            chosen = _resolve_choice(user_input, offered)
            if chosen:
                debug("recommendation_agent", f"user chose {chosen.get('item_name')!r}")
                return _reply_with_pick(
                    chosen, address_id, state=state,
                    awaiting_feedback=False, satisfied=True, last_recommendations=[chosen],
                )

        if len(offered) == 1 and _looks_like_decline_only_offer(user_input):
            restaurant_scope = recommendation_state.get("restaurant_scope")
            already_shown = {str(d.get("item_name") or "") for d in offered}
            if restaurant_scope and restaurant_scope.get("restaurant_id"):
                debug(
                    "recommendation_agent",
                    f"user declined the only match — widening within restaurant "
                    f"{restaurant_scope.get('restaurant_id')!r}",
                )
                widened_reply = _widen_within_restaurant(restaurant_scope, offered, address_id)
                if widened_reply:
                    return widened_reply
                return _reply(
                    "That's everything I can find at this restaurant beyond what's already in "
                    "your cart -- want to browse other restaurants instead?",
                    awaiting_feedback=False, satisfied=False, last_recommendations=offered,
                )

            debug("recommendation_agent", "user declined the only match — widening the search")
            widened = widen_recommendations(brief, address_id=address_id, exclude_items=already_shown)
            if widened:
                batch, rest = widened[:DISPLAY_COUNT], widened[DISPLAY_COUNT:]
                return _reply(
                    f"Here's what turns up with a wider search:\n{_numbered(batch)}\n{PICK_PROMPT}",
                    awaiting_feedback=True, satisfied=False,
                    last_recommendations=batch, alternates=rest, filters=brief_filters,
                    options=_dish_pick_options(batch),
                )
            return _reply(
                "That really was the only close match — tell me what to change (cuisine, "
                "budget, or dietary preference) and I'll look again.",
                awaiting_feedback=False, satisfied=False, last_recommendations=offered,
            )

        if _looks_like_more_request(user_input):
            debug("recommendation_agent", "user asked for more")
            return _serve_next_batch(
                "Here are a few more:", offered, alternates, brief_filters,
                recommendation_state.get("restaurant_scope"),
            )

        if _looks_like_acceptance(user_input):
            if len(offered) == 1:
                if is_category_offer:
                    return _handle_category_pick([offered[0]], recommendation_state, address_id)
                return _reply_with_pick(
                    offered[0], address_id, state=state,
                    awaiting_feedback=False, satisfied=True, last_recommendations=offered,
                )
            return _reply(
                f"Which one would you like — {' or '.join(str(i) for i in range(1, len(offered) + 1))}?",
                awaiting_feedback=True, satisfied=False,
                last_recommendations=offered, alternates=alternates, filters=brief_filters,
                options=_dish_pick_options(offered),
            )

        if _looks_like_rejection(user_input):
            debug("recommendation_agent", "user rejected the batch")
            if alternates:
                return _serve_next_batch(
                    "No problem, here are some others:", offered, alternates, brief_filters,
                    recommendation_state.get("restaurant_scope"),
                )
            return _reply(
                "No problem — tell me what you want to change, such as cuisine, budget, "
                "or dietary preference.",
                awaiting_feedback=False, satisfied=False, last_recommendations=offered,
            )

        if brief_filters == (recommendation_state.get("filters") or {}):
            debug("recommendation_agent", "unrecognised reply, filters unchanged — treating as 'more'")
            return _serve_next_batch(
                "Here are a few more:", offered, alternates, brief_filters,
                recommendation_state.get("restaurant_scope"),
            )

    ranked = fetch_recommendations(brief, address_id=address_id)
    if not ranked:
        return _reply(
            "I’m not finding a strong match right now. Tell me your cuisine, budget, "
            "or dietary preference and I’ll refine it.",
            awaiting_feedback=False, satisfied=False, last_recommendations=[],
        )

    batch, rest = ranked, []
    debug("recommendation_agent", f"offering all {len(batch)} match(es) as a scrollable list")
    return _reply(
        _build_recommendation_text(batch),
        awaiting_feedback=True, satisfied=False,
        last_recommendations=batch, alternates=rest, filters=brief_filters,
        options=ONLY_MATCH_OPTIONS if len(batch) == 1 else _dish_pick_options(batch),
    )
