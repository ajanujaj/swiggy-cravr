"""Recommendation agent: uses the MCP recommendation tool and a human-in-the-loop feedback cycle."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..cart.CartAgent import add_dish_to_cart, _current_restaurant_id, _current_restaurant_name
from ..client import call_tool_sync, using_real_food_mcp
from ..debug_utils import debug
from ..intent import KNOWN_CUISINE_WORDS, NON_VEG_ANSWERS, VEG_ANSWERS, extract_tastes
from ..profile_utils import latest_user_text
from ..state import CravrState

# Menu's `dietary_type` column only holds these values, compared with `==` --
# anything else (e.g. "vegetarian") silently matches zero rows.
DIETARY_TYPE_BY_PREFERENCE = {
    "veg": "Veg",
    "vegetarian": "Veg",
    "pure veg": "Veg",
    "veggie": "Veg",
    "vegan": "Veg",  # no pure-vegan dietary_type exists; closest faithful filter
    "non-veg": "Non-Veg",
    "non veg": "Non-Veg",
    "nonveg": "Non-Veg",
    "non-vegetarian": "Non-Veg",
    "non vegetarian": "Non-Veg",
    "eggetarian": "Non-Veg",
    "meat": "Non-Veg",
}

# Closed tag vocabulary in menu.csv -- anything else zeroes the result set.
KNOWN_TAGS = {
    "breakfast", "comfort-food", "dessert", "gluten-free", "high-fiber",
    "high-protein", "lightweight", "low-calorie", "low-carb", "mild",
    "one-pot-meal", "organic", "savory", "spicy", "sugar-free-option",
    "sweet", "tangy", "vegan", "vegan-option", "vegetarian",
}

# Matched against menu.csv's `course` column; order matters, first hit wins.
BEVERAGE_CUES = ("coffee", "tea", "chai", "drink", "beverage", "juice", "smoothie",
                 "latte", "cappuccino", "espresso", "shake", "lassi")
DESSERT_CUES = ("dessert", "sweet dish", "pudding", "cake", "brownie", "ice cream",
                "sweet tooth", "something sweet", "pastry")
SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "small plate",
             "nibble", "finger food")

MEAL_COURSES = "main,side"  # fallback, not a first choice -- see _course_preference

# How many dishes to put in front of the user at once.
DISPLAY_COUNT = 3

ACCEPTANCE_TOKENS = ("yes", "yep", "yeah", "sure", "ok", "okay", "perfect",
                     "great", "sounds good", "looks good", "love it",
                     "go ahead", "that works", "book it", "accept")
REJECTION_TOKENS = ("no", "nope", "nah", "not that", "different", "another",
                    "change", "new one", "something else", "skip", "reject")
# Distinct from rejection -- "more" isn't unhappy with the offer, just wants the rest.
MORE_TOKENS = ("more", "other", "others", "else", "next", "alternatives",
               "alternative", "rest", "again")


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    return latest_user_text(messages)


def _mentions(user_input: str, tokens: tuple[str, ...]) -> bool:
    """Whole-word/phrase match -- plain substring matching read "I want
    noodles" as a rejection (contains "no") and "not good" as acceptance
    (contains "good")."""
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
    """Offer up to three numbered options -- one forces a yes/no on a
    stranger's taste; show fewer rather than padding with bad fits."""
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

# Numbered so the single-match case also costs one keystroke, not a typed-out yes/no.
ONLY_MATCH_PROMPT = "    1. Yes, I'll take it\n    2. Widen the search"


def _numbered(dishes: list[dict[str, Any]]) -> str:
    return "\n".join(f"  {i}. {_describe(d)}" for i, d in enumerate(dishes, start=1))


def _dish_pick_options(dishes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Tappable buttons for a numbered dish list, matching `_numbered`'s
    order. Values are the bare position ("1", "2", ...) -- same as what a
    typed digit resolves to via `_resolve_choice` -- so button and typed
    reply take the same path."""
    return [
        {"value": str(i), "label": d.get("item_name") or f"Option {i}"}
        for i, d in enumerate(dishes, start=1)
    ]


# Single source of truth for ONLY_MATCH_PROMPT's two lines, so UI buttons can't drift out of sync.
ONLY_MATCH_OPTIONS = [
    {"value": "1", "label": "Yes, I'll take it"},
    {"value": "2", "label": "Widen the search"},
]


# Order matters: cardinals checked first would read "the third one"'s "one" as option 1.
ORDINAL_WORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}
CARDINAL_WORDS = {"one": 1, "two": 2, "three": 3}


def _resolve_choice(user_input: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Work out which offered dish the user meant: "2", "the second one",
    "#3", "the biryani". Checked before yes/no parsing, so "no, the second
    one" selects rather than rejects."""
    normalized = (user_input or "").strip().lower()
    if not normalized or not options:
        return None

    for word, position in ORDINAL_WORDS.items():
        if position <= len(options) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized):
            return options[position - 1]

    # Bare cardinals only count when they ARE the message or follow "number"/"#" --
    # otherwise they're usually filler ("the thai one").
    for word, position in CARDINAL_WORDS.items():
        if position > len(options):
            continue
        if normalized.strip(" .!") == word or re.search(
            rf"(?:number|option|#)\s*{re.escape(word)}(?!\w)", normalized
        ):
            return options[position - 1]

    # A fresh search offers every match at once (recommendation_agent_node's
    # "offering all N match(es) as a scrollable list"), so a pick can be a
    # two/three-digit position ("10", "36") -- not just 1-9. Bounded to
    # len(options) so a genuinely out-of-range number ("under 200") still
    # falls through to _carries_new_constraints' price heuristic instead of
    # being misread as a pick.
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", normalized)
    if match and 1 <= int(match.group(1)) <= len(options):
        return options[int(match.group(1)) - 1]

    # Require a distinctive word so a stray "rice"/"veg" doesn't misfire.
    for dish in options:
        words = [w for w in re.findall(r"\w+", str(dish.get("item_name") or "").lower()) if len(w) >= 5]
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", normalized) for w in words):
            return dish

    return None


def _course_preference(intent_profile: dict[str, Any]) -> str:
    """Infer which `course` the user wants; defaults to "main" -- an open
    field lets "something spicy" match a ₹70 Filter Coffee."""
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
    # Weaker signal than the word "dessert", but enough to include sweet courses.
    if "sweet" in [str(t).lower() for t in intent_profile.get("taste_preference") or []]:
        return "dessert,main"
    if any(cue in haystack for cue in SIDE_CUES):
        return "side"
    return "main"


def _build_recommendation_filters(intent_profile: dict[str, Any]) -> dict[str, Any]:
    """Build the MCP `recommendation` tool's argument dict from
    intent_profile. Only forward keys the tool accepts (cuisine, tags,
    dietary_type, max_price_inr, ...) -- anything else fails validation.

    Two rules: `cuisine` never goes into `tags` (no row tags "Indian",
    would zero the result); `dietary_type` must use the menu's own
    vocabulary ("Veg"/"Non-Veg"), compared exactly.
    """
    filters: dict[str, Any] = {}

    tag_terms: list[str] = []
    for term in [intent_profile.get("craving"), intent_profile.get("craving_tag"), *(intent_profile.get("taste_preference") or [])]:
        if not term:
            continue
        normalized = str(term).strip().lower()
        # Only real tags survive -- "biryani" isn't a tag, would zero the result.
        if normalized in KNOWN_TAGS and normalized not in tag_terms:
            tag_terms.append(normalized)
    if tag_terms:
        # Comma-separated tags are AND-matched by the MCP recommendation tool.
        filters["tags"] = ",".join(tag_terms)

    # The exact dish/type name the user picked or typed -- kept verbatim (not filtered
    # through KNOWN_TAGS) so a real-MCP search can query Swiggy's search_menu tool
    # directly by name (e.g. "sushi"), which isn't a menu.csv tag at all.
    if intent_profile.get("craving"):
        filters["craving"] = str(intent_profile["craving"])

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
    """Boil order history down to what actually informs a pick: which
    cuisines the user keeps coming back to, and which items repeat."""
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
    """One concise line covering intent + history, for logging and display."""
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
    """Apply durable defaults only where this request did not specify one."""
    merged = dict(intent_profile)
    preferences = _coerce_profile(preference_profile).get("preferences") or {}
    if isinstance(preferences, list):  # older MCP shape: a list instead of one profile
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
    """Freeze a request into everything the recommender needs, and nothing
    more. Called once by decision_agent so the profile-to-filter translation
    happens exactly once, not re-derived downstream."""
    effective_intent = _apply_saved_preferences(
        _coerce_profile(intent_profile), preference_profile
    )
    filters = _build_recommendation_filters(effective_intent)
    history = _summarise_history(_coerce_profile(history_profile))
    return {"filters": filters, "history": history, "summary": _brief_summary(filters, history)}


def _filter_variants(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Strict filters first, then progressively looser fallbacks.
    `dietary_type`/`max_price_inr` are hard constraints, never relaxed.
    Cuisine is dropped before tags -- taste is usually what the user said
    out loud, cuisine is often our own inference.
    """
    # Widen course (main -> main,side) rather than drop it -- dropping it is
    # how a beverage answers "something spicy"; a spicy side beats a mild main.
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


def _rank_candidates(
    recommendations: list[dict[str, Any]],
    desired_tags: list[str],
    desired_cuisine: str | None,
    desired_course: str | None,
    history: dict[str, Any],
) -> list[dict[str, Any]]:
    """Order results by how well they honour the *original* request, not
    just rating -- rating-only sorting once recommended a ₹70 Filter Coffee
    to someone who asked for something spicy, since it had the best rating
    in the relaxed set."""
    wanted_tags = {t.strip().lower() for t in desired_tags if t}
    cuisine = (desired_cuisine or "").strip().lower()
    # First listed course is the one actually wanted; rest are widened fallbacks.
    preferred_course = (desired_course or "").split(",")[0].strip().lower()

    past_cuisines = {str(c).strip().lower() for c in (history or {}).get("top_cuisines") or []}

    def sort_key(item: dict[str, Any]) -> tuple[int, int, int, int, float]:
        item_tags = {t.strip().lower() for t in str(item.get("tags") or "").split(";") if t.strip()}
        item_cuisine = str(item.get("cuisine") or "").strip().lower()
        item_course = str(item.get("course") or "").strip().lower()
        try:
            rating = float(item.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0.0
        return (
            1 if preferred_course and item_course == preferred_course else 0,  # a meal, not a side
            len(wanted_tags & item_tags),                          # asked-for taste
            1 if cuisine and cuisine in item_cuisine else 0,       # then cuisine
            1 if item_cuisine in past_cuisines else 0,             # then familiarity
            rating,                                                # then quality
        )

    return sorted(recommendations, key=sort_key, reverse=True)

def _unwrap_tool_result(result: Any) -> list[dict[str, Any]]:
    """Pull the recommendations list out of whatever shape fastmcp returns."""
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
    """Fallback lookup for a direct/test invocation that skips
    recommendation_agent_node (which normally passes the address already
    resolved by context_agent). Deliberately NOT cached across calls -- an
    old `functools.lru_cache(maxsize=1)` here memoized the *first* result,
    success or failure, for the whole process, so one early failure sent
    `addressId: null` forever after.
    """
    if address_id:
        return address_id
    return food_mcp.get_saved_address().get("address_id")


def _call_recommendation_tool(filters: dict[str, Any], address_id: str | None = None) -> list[dict[str, Any]]:
    if using_real_food_mcp():
        return food_mcp.search_dishes(filters, address_id=_resolve_address_id(address_id))

    # The mock `recommendation` tool's signature has no `craving` parameter --
    # FastMCP rejects an unexpected argument, so this would otherwise fail every
    # call once a specific dish/type is known. Real MCP uses it (food_mcp.search_dishes
    # queries Swiggy's search_menu directly by name); mock mode just drops it.
    mock_filters = {k: v for k, v in filters.items() if k != "craving"}
    try:
        result = call_tool_sync("recommendation", mock_filters)
    except Exception as exc:
        # Distinguishes "MCP server down" from "no dish matched" -- both used to
        # surface as the same shrug.
        debug("recommendation_agent", f"MCP call FAILED ({type(exc).__name__}: {exc})")
        return []

    return _unwrap_tool_result(result)


def fetch_recommendations(
    brief: dict[str, Any], address_id: str | None = None, exclude_restaurant_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Call the MCP recommendation tool, relaxing filters until something
    matches. Takes only the frozen brief from decision_agent, plus the
    address already resolved by context_agent. `exclude_restaurant_ids` is
    for cart_agent's "browse other restaurants" -- drop whatever's already
    in (or was just cleared from) the cart so it can't reappear as its own
    suggestion."""
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
                recommendations, desired_tags, desired_cuisine, desired_course, history
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
    """Relax the search beyond the strict match, for when
    `fetch_recommendations` found only one result and the user wants to see
    more. Reuses `_filter_variants`, starting one step past the strict
    variant already tried."""
    brief = _coerce_profile(brief)
    filters = dict(brief.get("filters") or {})
    history = _coerce_profile(brief.get("history"))
    exclude = {str(name).strip().lower() for name in (exclude_items or set())}

    desired_tags = [t for t in str(filters.get("tags") or "").split(",") if t]
    desired_cuisine = filters.get("cuisine")
    desired_course = filters.get("course")

    # [0] already produced the single result shown; start at the first relaxed variant.
    for attempt, variant in enumerate(_filter_variants(filters)[1:], start=1):
        recommendations = _call_recommendation_tool(variant, address_id=address_id)
        recommendations = [
            item for item in recommendations
            if str(item.get("item_name") or "").strip().lower() not in exclude
        ]
        debug("recommendation_agent", f"widen attempt {attempt} filters={variant} -> {len(recommendations)} result(s)")
        if recommendations:
            return _rank_candidates(recommendations, desired_tags, desired_cuisine, desired_course, history)

    debug("recommendation_agent", "widen found nothing beyond the original match")
    return []


def _carries_new_constraints(user_input: str) -> bool:
    """True when the message changes the request rather than just reacting to it."""
    normalized = (user_input or "").strip().lower()
    # Two or more digits is a price, not a menu position ("under 200").
    if re.search(r"\d{2,}", normalized):
        return True
    if extract_tastes(normalized):
        return True
    if any(word in normalized for word in KNOWN_CUISINE_WORDS):
        return True
    return _mentions(normalized, VEG_ANSWERS + NON_VEG_ANSWERS)


# "2" means "widen" only for ONLY_MATCH_PROMPT's two-choice case; against a
# real dish list "2" means "the second one" (handled by `_resolve_choice`).
WIDEN_TOKENS = ("widen", "broaden", "expand", "wider")


def _looks_like_decline_only_offer(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower().rstrip(".!")
    if normalized == "2":
        return True
    return _looks_like_rejection(user_input) or _mentions(user_input, WIDEN_TOKENS)


def is_pure_feedback(user_input: str, recommendation_state: dict[str, Any] | None) -> bool:
    """True when a message only reacts to the dishes just offered ("3",
    "yes") -- lets the graph skip intent/history/decision, and avoids
    feeding "3" to the extraction model as budget_inr=3. Conservative:
    anything carrying a new constraint ("make it non-veg") goes the long
    way round instead.
    """
    state = recommendation_state or {}
    offered = state.get("last_recommendations") or []
    if not offered or not state.get("awaiting_feedback") or state.get("satisfied"):
        return False

    # An explicit pick is unambiguous even if it collides with other vocabulary.
    if _resolve_choice(user_input, offered):
        return True
    # Same reasoning for the single-match "2" (see ONLY_MATCH_PROMPT) -- as
    # unambiguous as a real pick, so it also skips the new-constraints gate.
    # A typed-out "widen the search" isn't as unambiguous, so it still goes
    # the long way round below.
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
    """Terminal "user picked a dish" reply. Auto-adds it to the real cart --
    picking isn't spending money (place_food_order's hard confirmation gate
    is where that risk actually lives), so no extra ask here, just add and
    show the cart. Goes through cart.CartAgent.add_dish_to_cart so this and
    "add it to cart" share one code path. `state`, when the caller has it,
    lets add_dish_to_cart catch a cross-restaurant add before it reaches
    the real API -- live evidence: merging an item from a different
    restaurant into a non-empty cart made update_food_cart wipe the whole
    cart rather than just fail to add the new item.
    """
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
) -> dict:
    """Show the next slice of an already-fetched result set -- `alternates`
    holds the rest so "show me more" doesn't re-run the query and re-print
    the same three dishes."""
    if alternates:
        batch, rest = alternates[:DISPLAY_COUNT], alternates[DISPLAY_COUNT:]
        debug("recommendation_agent", f"serving {len(batch)} more, {len(rest)} still held back")
        return _reply(
            f"{lead_in}\n{_numbered(batch)}\n{PICK_PROMPT}",
            awaiting_feedback=True, satisfied=False,
            last_recommendations=batch, alternates=rest, filters=filters,
            options=_dish_pick_options(batch),
        )

    debug("recommendation_agent", "no alternates left to offer")
    return _reply(
        "That's everything on the menu matching what you asked for. I can widen it — "
        "a different cuisine, a higher budget, or including sides and desserts?",
        awaiting_feedback=True, satisfied=False,
        last_recommendations=offered, alternates=[], filters=filters,
    )


def recommendation_agent_node(state: CravrState) -> dict:
    """Recommend food via MCP and keep looping until the user is satisfied.
    Reads the frozen `recommendation_brief` from decision_agent; the
    fallback below only matters for a direct/test invoke outside the graph.
    """
    brief = _coerce_profile(state.get("recommendation_brief"))
    if not brief:
        debug("recommendation_agent", "no brief in state — deriving one from the profiles")
        brief = build_recommendation_brief(
            _coerce_profile(state.get("intent_profile")),
            _coerce_profile(state.get("history_profile")),
        )

    brief_filters = dict(brief.get("filters") or {})
    address_id = _coerce_profile(state.get("context_profile")).get("address_id")  # resolved per-turn by context_agent
    recommendation_state = dict(state.get("recommendation_state") or {})
    awaiting_feedback = bool(recommendation_state.get("awaiting_feedback"))
    offered = recommendation_state.get("last_recommendations") or []
    alternates = recommendation_state.get("alternates") or []
    satisfied = bool(recommendation_state.get("satisfied"))

    user_input = _extract_latest_user_input(state.get("messages") or [])
    debug("recommendation_agent", f"user_input={user_input!r} awaiting_feedback={awaiting_feedback}")

    if awaiting_feedback and not satisfied:
        # "no, the second one" is a choice, not a rejection -- check picks first.
        chosen = _resolve_choice(user_input, offered)
        if chosen:
            debug("recommendation_agent", f"user chose {chosen.get('item_name')!r}")
            return _reply_with_pick(
                chosen, address_id, state=state,
                awaiting_feedback=False, satisfied=True, last_recommendations=[chosen],
            )

        # A single-match "2"/decline/widen broadens the search itself, unlike
        # the generic rejection branch below -- checked first.
        if len(offered) == 1 and _looks_like_decline_only_offer(user_input):
            debug("recommendation_agent", "user declined the only match — widening the search")
            already_shown = {str(d.get("item_name") or "") for d in offered}
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
            return _serve_next_batch("Here are a few more:", offered, alternates, brief_filters)

        if _looks_like_acceptance(user_input):
            if len(offered) == 1:
                return _reply_with_pick(
                    offered[0], address_id, state=state,
                    awaiting_feedback=False, satisfied=True, last_recommendations=offered,
                )
            # Ambiguous against a list of three -- don't guess which one.
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
                    "No problem, here are some others:", offered, alternates, brief_filters
                )
            return _reply(
                "No problem — tell me what you want to change, such as cuisine, budget, "
                "or dietary preference.",
                awaiting_feedback=False, satisfied=False, last_recommendations=offered,
            )

        if brief_filters == (recommendation_state.get("filters") or {}):
            # Unrecognised reply, but filters unchanged -- re-running would
            # return the same list, so show what we're already holding.
            debug("recommendation_agent", "unrecognised reply, filters unchanged — treating as 'more'")
            return _serve_next_batch("Here are a few more:", offered, alternates, brief_filters)

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
