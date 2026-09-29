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
from .signals import attach_restaurant_info, history_signals, practical_score
from ..userPreference.taste import record_pick, record_rejection, taste_signals
from . import feedback
from .reasons import with_reasons
from .opening import OPENING_KIND, SOMETHING_ELSE_LABEL, SOMETHING_ELSE_VALUE
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
    if dish.get("eta_min"):
        details.append(f"{dish['eta_min']} min")
    if dish.get("offer"):
        details.append(str(dish["offer"]))
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
    return "\n".join(
        f"  {i}. {_describe(d)}" + (f" — {d['reason']}" if d.get("reason") else "")
        for i, d in enumerate(dishes, start=1)
    )


def _dish_pick_options(dishes: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"value": str(i), "label": d.get("item_name") or f"Option {i}"}
        for i, d in enumerate(dishes, start=1)
    ]


def _dish_multiselect_options(dishes: list[dict[str, Any]]) -> list[dict[str, str]]:
    options = []
    for i, dish in enumerate(dishes, start=1):
        name = dish.get("item_name") or f"Option {i}"
        price = dish.get("price_inr")
        label = f"{name} — ₹{price:.0f}" if isinstance(price, (int, float)) and price else name
        options.append({"value": str(i), "label": label})
    return options


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

    for dish in options:
        if str(dish.get("item_name") or "").strip().lower() == normalized:
            return dish

    for dish in options:
        words = [w for w in re.findall(r"\w+", str(dish.get("item_name") or "").lower()) if len(w) >= 5]
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", normalized) for w in words):
            return dish

    return None


_NO_SELECTION_VALUE = "__no_addon__"

# A multi-select token can carry an optional quantity marker so a single
# turn can say "2 of these, 1 of that" instead of everything defaulting to
# one. Three shapes are recognised, tried in this order:
#   "3:2"              -- chip-submitted (OptionChips.jsx's quantity
#                          stepper encodes a >1 pick as "value:qty")
#   "Falafel Wrap x2"   -- typed suffix, also covers "3x2" (index 3, qty 2)
#   "2x Falafel Wrap"   -- typed prefix (guarded so it never also matches
#                          the "3x2" suffix shape -- the remainder after
#                          the marker must not itself be a bare number)
# No marker at all still means quantity 1, exactly as before.
_QTY_COLON_RE = re.compile(r"^(.*\S)\s*:\s*(\d{1,2})\s*$")
_QTY_XSUFFIX_RE = re.compile(r"^(.*\S)\s*[x\u00d7]\s*(\d{1,2})\s*$", re.IGNORECASE)
_QTY_XPREFIX_RE = re.compile(r"^\s*(\d{1,2})\s*[x\u00d7]\s*(\D.*\S)\s*$", re.IGNORECASE)


def _split_quantity(token: str) -> tuple[str, int]:
    t = (token or "").strip()
    for pattern in (_QTY_COLON_RE, _QTY_XSUFFIX_RE):
        m = pattern.match(t)
        if m:
            return m.group(1).strip(), max(1, min(20, int(m.group(2))))
    m = _QTY_XPREFIX_RE.match(t)
    if m:
        return m.group(2).strip(), max(1, min(20, int(m.group(1))))
    return t, 1


def _resolve_multiple_choices(
    user_input: str, options: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], int]]:
    if not (user_input or "").strip() or not options:
        return []

    tokens = [t.strip() for t in re.split(r",|\band\b|&", user_input, flags=re.IGNORECASE) if t.strip()]
    chosen: list[tuple[dict[str, Any], int]] = []
    seen_ids: set[int] = set()
    for token in tokens:
        identifier, quantity = _split_quantity(token)
        match = _resolve_choice(identifier, options)
        if match is not None and id(match) not in seen_ids:
            seen_ids.add(id(match))
            chosen.append((match, quantity))
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


BUDGET_FRIENDLY_MAX_INR = 200
_BUDGET_CUES = ("budget", "cheap", "pocket", "affordable", "value")


def _is_budget_category(craving_tag: Any) -> bool:
    text = str(craving_tag or "").strip().lower()
    return bool(text) and any(cue in text for cue in _BUDGET_CUES)


def _free_text_category(craving_tag: Any) -> str | None:
    """The first-round answer when it names food, not a mood tag, cuisine or budget."""
    text = str(craving_tag or "").strip()
    lowered = text.lower()
    if (
        len(lowered) < 2
        or lowered.isdigit()
        or text in CRAVING_TAG_CUES
        or lowered in KNOWN_TAGS
        or lowered in KNOWN_CUISINE_WORDS
        or _is_budget_category(lowered)
    ):
        return None
    return text


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
        if (
            craving_tag
            and len(str(craving_tag).split()) >= 2
            and str(craving_tag) not in CRAVING_TAG_CUES
            and str(craving_tag).strip().lower() not in KNOWN_CUISINE_WORDS
            and str(craving_tag).strip().lower() != craving.strip().lower()
        ):
            filters["craving_secondary"] = str(craving_tag)
    else:
        # No craving, but the first answer was free text ("Small wrap", "Keto") rather
        # than a known mood tag -- that's the only thing the user told us, so search for
        # it instead of dropping it (it used to fall through to a generic "restaurants" search).
        free_text = _free_text_category(intent_profile.get("craving_tag"))
        if free_text:
            filters["craving"] = free_text

    cuisine = intent_profile.get("cuisine")
    if not cuisine and str(intent_profile.get("craving_tag") or "").strip().lower() in KNOWN_CUISINE_WORDS:
        cuisine = intent_profile["craving_tag"]
    if cuisine:
        filters["cuisine"] = str(cuisine)

    filters["course"] = _course_preference(intent_profile)

    for preference in intent_profile.get("dietary_preferences") or []:
        dietary_type = DIETARY_TYPE_BY_PREFERENCE.get(str(preference).strip().lower())
        if dietary_type:
            filters["dietary_type"] = dietary_type
            break

    budget = intent_profile.get("budget_inr")
    if budget is None and _is_budget_category(intent_profile.get("craving_tag")):
        budget = BUDGET_FRIENDLY_MAX_INR
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
    restaurants = Counter(
        str(o.get("restaurant_name")).strip() for o in orders if o.get("restaurant_name")
    )
    # Real Swiggy orders carry every item in `items`; mock orders only `item_name`.
    items: Counter = Counter()
    for o in orders:
        names = {str(i.get("name")).strip() for i in (o.get("items") or []) if i.get("name")}
        if not names and o.get("item_name"):
            names = {str(o["item_name"]).strip()}
        items.update(names)

    return {
        "order_count": len(orders),
        "top_cuisines": [name for name, _count in cuisines.most_common(2)],
        "top_restaurants": [name for name, _count in restaurants.most_common(3)],
        "repeat_items": [name for name, count in items.most_common(3) if count > 1],
        # Counts + recency used by signals.practical_score when ranking.
        **history_signals(history_profile),
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
    if history.get("top_restaurants"):
        parts.append("often orders from " + ", ".join(history["top_restaurants"]))
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
    history = {
        **_summarise_history(_coerce_profile(history_profile)),
        **taste_signals(preference_profile),  # learned likes/dislikes (feature 1)
    }
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


def _ranking_signals(history: dict[str, Any] | None) -> dict[str, Any]:
    """Signals for practical_score. Briefs built before signals existed (or
    tests) only carry top_cuisines/top_restaurants -- treat those as counts of 1."""
    history = dict(history or {})
    if "restaurant_counts" not in history:
        history["restaurant_counts"] = {str(r).strip().lower(): 1 for r in history.get("top_restaurants") or []}
    if "cuisine_counts" not in history:
        history["cuisine_counts"] = {str(c).strip().lower(): 1 for c in history.get("top_cuisines") or []}
    return history


def _rank_candidates(
    recommendations: list[dict[str, Any]],
    desired_tags: list[str],
    desired_cuisine: str | None,
    desired_course: str | None,
    history: dict[str, Any],
    desired_craving: str | None = None,
    desired_craving_secondary: str | None = None,
) -> list[dict[str, Any]]:
    """Relevance first (what they asked for), then one practical score:
    their usual restaurants/dishes, delivery time, offer, rating, and a
    penalty for what they ate in the last day (see signals.py)."""
    wanted_tags = {t.strip().lower() for t in desired_tags if t}
    cuisine = (desired_cuisine or "").strip().lower()
    preferred_course = (desired_course or "").split(",")[0].strip().lower()
    signals = _ranking_signals(history)

    def sort_key(item: dict[str, Any]) -> tuple[int, int, int, int, float]:
        item_tags = {t.strip().lower() for t in str(item.get("tags") or "").split(";") if t.strip()}
        item_cuisine = str(item.get("cuisine") or "").strip().lower()
        item_course = str(item.get("course") or "").strip().lower()
        return (
            _craving_match_score(
                str(item.get("item_name") or ""), desired_craving, desired_craving_secondary
            ),
            1 if preferred_course and item_course == preferred_course else 0,
            len(wanted_tags & item_tags),
            1 if cuisine and cuisine in item_cuisine else 0,
            practical_score(item, signals),
        )

    return sorted(recommendations, key=sort_key, reverse=True)


# Restaurants looked up per search for delivery time / offer / open status.
_ENRICH_RESTAURANTS = 5


def _enrich_with_restaurant_info(
    recommendations: list[dict[str, Any]], address_id: str | None,
) -> list[dict[str, Any]]:
    """Real Swiggy only: add delivery time, offer and open status from
    `search_restaurants` for the first few restaurants in the result, and
    drop dishes from restaurants that are closed."""
    if not recommendations or not using_real_food_mcp():
        return recommendations
    wanted: list[tuple[str, str]] = []
    for dish in recommendations:
        pair = (str(dish.get("restaurant_id") or ""), str(dish.get("restaurant_name") or ""))
        if pair[0] and pair[1] and pair not in wanted:
            wanted.append(pair)
        if len(wanted) >= _ENRICH_RESTAURANTS:
            break
    try:
        info = food_mcp.restaurant_info(wanted, _resolve_address_id(address_id))
    except Exception as exc:  # never lose results over missing extras
        debug("recommendation_agent", f"restaurant info lookup failed ({exc}) -- ranking without it")
        return recommendations
    enriched = attach_restaurant_info(recommendations, info)
    dropped = len(recommendations) - len(enriched)
    if dropped:
        debug("recommendation_agent", f"dropped {dropped} dish(es) from closed restaurants")
    return enriched


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
            recommendations = _enrich_with_restaurant_info(recommendations, address_id)
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
            return with_reasons(ranked, _ranking_signals(history), filters)

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
            recommendations = _enrich_with_restaurant_info(recommendations, address_id)
            return with_reasons(_rank_candidates(
                recommendations, desired_tags, desired_cuisine, desired_course, history,
                desired_craving=filters.get("craving"),
                desired_craving_secondary=filters.get("craving_secondary"),
            ), _ranking_signals(history), filters)

    debug("recommendation_agent", "widen found nothing beyond the original match")
    return []


def _offer_category_batch(
    lead_in: str, categories: list[dict[str, Any]], restaurant_scope: dict[str, Any] | None = None,
) -> dict:
    summaries = [{"item_name": c["category"], "category": c["category"]} for c in categories]
    lines = "\n".join(
        f"  {i}. {c['category']} ({len(c['items'])} item{'' if len(c['items']) == 1 else 's'})"
        for i, c in enumerate(categories, start=1)
    )
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
    content = (
        f"{lead_in}\n{_numbered(items)}\nPick as many as you like, then hit submit."
        if lead_in else _numbered(items)
    )
    return _reply(
        content,
        awaiting_feedback=True, satisfied=False,
        last_recommendations=items, alternates=[], filters=filters or {},
        offer_kind="dish",
        options=_dish_multiselect_options(items),
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


def _handle_dish_multiselect(
    dishes: list[dict[str, Any]], address_id: str | None, state: CravrState | None,
    quantities: dict[str, int] | None = None,
) -> dict:
    if not dishes:
        return _reply("Nothing was picked to add.", awaiting_feedback=False, satisfied=False)

    if not address_id:
        names = _join_labels([_describe(d) for d in dishes])
        return _reply(
            f"Great choices — {names} — but I don't have a delivery address yet, so I "
            "couldn't add them to your cart.",
            awaiting_feedback=False, satisfied=False,
        )

    known_restaurant_id = _current_restaurant_id(state) if state is not None else None
    known_restaurant_name = _current_restaurant_name(state) if state is not None else None

    added: list[str] = []
    result: dict[str, Any] = {}
    for dish in dishes:
        quantity = 1
        if quantities:
            quantity = max(1, int(quantities.get(str(dish.get("menu_item_id")), 1) or 1))
        result = add_dish_to_cart(
            dish, address_id, quantity=quantity,
            known_restaurant_id=known_restaurant_id, known_restaurant_name=known_restaurant_name,
        )
        cart_state = result.get("cart_state") or {}
        if "restaurant" not in cart_state:
            if added:
                prior_note = f"Added {_join_labels(added)} first. "
                messages = result.get("messages") or []
                original = getattr(messages[0], "content", "") if messages else ""
                result = {**result, "messages": [AIMessage(content=prior_note + original, name="cart_agent")]}
            result["recommendation_state"] = {"awaiting_feedback": False, "satisfied": False}
            return result
        label = dish.get("item_name") or "that dish"
        added.append(f"{quantity}x {label}" if quantity > 1 else label)
        restaurant = cart_state["restaurant"]
        known_restaurant_id = known_restaurant_id or restaurant.get("id")
        known_restaurant_name = known_restaurant_name or restaurant.get("name")

    messages = result.get("messages") or []
    body = getattr(messages[0], "content", "") if messages else ""
    lead_in = f"Added {_join_labels(added)} to your cart."
    content = f"{lead_in}\n\n{body}".strip() if body else lead_in
    return {
        **result,
        "messages": [AIMessage(content=content, name="cart_agent")],
        "recommendation_state": {
            "awaiting_feedback": False, "satisfied": True, "last_recommendations": dishes,
        },
    }


def _widen_within_restaurant(
    restaurant_scope: dict[str, Any], offered: list[dict[str, Any]], address_id: str | None,
) -> dict | None:
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

    if (user_input or "").strip() == _NO_SELECTION_VALUE:
        return True

    if state.get("offer_kind") == OPENING_KIND:
        # Only a pick is a reaction to the opening cards. "Something else", "no",
        # "more" or anything typed is a (new) request -- intent asks what they want.
        return bool(_resolve_choice(user_input, offered))

    if _resolve_choice(user_input, offered):
        return True
    if len(offered) == 1 and (user_input or "").strip().rstrip(".!") == "2":
        return True
    # "too pricey", "something lighter", "too spicy": a reason for rejecting what's shown
    # (feature 2). Checked before new constraints -- "too spicy" is not a request for spicy.
    if feedback.infer_reason(user_input) and not re.search(r"\d{2,}", user_input or ""):
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
    record_pick((state or {}).get("user_id"), chosen)  # feature 1: learn from every pick
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


def _handle_rejection(
    reason: str, state: CravrState, brief: dict[str, Any], rec_state: dict[str, Any], address_id: str | None,
) -> dict:
    """Fix what the user said was wrong (feature 2): tighten the search for that
    reason, remember it, and show a new batch -- from the results already held
    when enough of them pass, otherwise from a fresh search."""
    offered = rec_state.get("last_recommendations") or []
    alternates = rec_state.get("alternates") or []
    record_rejection(state.get("user_id"), offered, reason)

    adjustments = feedback.adjustments_for(reason, offered, rec_state.get("adjustments"))
    shown = {str(n).strip().lower() for n in rec_state.get("shown_items") or []}
    shown |= {str(d.get("item_name") or "").strip().lower() for d in offered}
    debug("recommendation_agent", f"rejection reason={reason!r} adjustments={adjustments}")

    def usable(dishes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out = []
        for d in dishes:
            key = str(d.get("item_name") or "").strip().lower()
            if key in shown or key in seen or not feedback.keeps(d, adjustments):
                continue
            seen.add(key)
            out.append(d)
        return out

    new_brief = feedback.adjusted_brief(brief, adjustments)
    candidates = usable(alternates)
    if len(candidates) < DISPLAY_COUNT:
        candidates = usable(candidates + fetch_recommendations(new_brief, address_id=address_id))

    filters = dict(new_brief.get("filters") or {})
    candidates = sorted(candidates, key=lambda d: feedback.preference_key(d, adjustments), reverse=True)
    if not candidates:
        return _reply(
            f"{feedback.NOTHING_FOUND.get(reason, 'Nothing else fits.')} Tell me what to change "
            "(a dish, cuisine or budget) and I'll look again.",
            awaiting_feedback=False, satisfied=False, last_recommendations=offered,
            adjustments=adjustments, shown_items=sorted(shown),
        )

    batch, rest = candidates[:DISPLAY_COUNT], candidates[DISPLAY_COUNT:]
    return _reply(
        f"{feedback.LEAD_INS.get(reason, 'Here are some others:')}\n{_numbered(batch)}\n{PICK_PROMPT}",
        awaiting_feedback=True, satisfied=False,
        last_recommendations=batch, alternates=rest, filters=filters,
        options=_dish_pick_options(batch),
        adjustments=adjustments, shown_items=sorted(shown),
    )


OPENING_LEAD = "Not sure what to eat? Here are three ideas to start with:"
OPENING_PROMPT = "Pick one, or tap Something else and I'll ask what you're in the mood for."


def _offer_opening(picks: list[dict[str, Any]], filters: dict[str, Any], token: str | None) -> dict:
    lines = []
    for i, dish in enumerate(picks, start=1):
        badge = f"{dish['badge']}: " if dish.get("badge") else ""
        reason = f" — {dish['reason']}" if dish.get("reason") else ""
        lines.append(f"  {i}. {badge}{_describe(dish)}{reason}")
    options = _dish_pick_options(picks) + [{"value": SOMETHING_ELSE_VALUE, "label": SOMETHING_ELSE_LABEL}]
    return _reply(
        f"{OPENING_LEAD}\n" + "\n".join(lines) + f"\n{OPENING_PROMPT}",
        awaiting_feedback=True, satisfied=False,
        last_recommendations=picks, alternates=[], filters=filters,
        options=options, offer_kind=OPENING_KIND, opening_token=token,
    )


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

    opening_token = brief.get("opening_token")
    showing_this_opening = (
        recommendation_state.get("offer_kind") == OPENING_KIND
        and opening_token
        and recommendation_state.get("opening_token") == opening_token
    )
    if brief.get("opening_picks") and not showing_this_opening:
        return _offer_opening(brief["opening_picks"], brief_filters, opening_token)
    if recommendation_state.get("offer_kind") == OPENING_KIND and not showing_this_opening:
        # They passed on the opening cards and told us what they want -- a fresh search.
        awaiting_feedback = False

    if awaiting_feedback and not satisfied:
        is_category_offer = recommendation_state.get("offer_kind") == "category"
        is_dish_multiselect_offer = recommendation_state.get("offer_kind") == "dish"
        if is_category_offer or is_dish_multiselect_offer:
            if user_input.strip() == _NO_SELECTION_VALUE:
                debug("recommendation_agent", "user skipped every option in the multi-select")
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
                if is_category_offer:
                    picked_categories = [dish for dish, _qty in chosen_many]
                    debug(
                        "recommendation_agent",
                        f"user picked {len(picked_categories)} categor"
                        f"{'y' if len(picked_categories) == 1 else 'ies'}: "
                        f"{[c.get('category') for c in picked_categories]}",
                    )
                    return _handle_category_pick(picked_categories, recommendation_state, address_id)
                picked_dishes = [dish for dish, _qty in chosen_many]
                quantities = {str(dish.get("menu_item_id")): qty for dish, qty in chosen_many}
                debug(
                    "recommendation_agent",
                    f"user picked {len(chosen_many)} dish(es): "
                    f"{[(dish.get('item_name'), qty) for dish, qty in chosen_many]}",
                )
                return _handle_dish_multiselect(picked_dishes, address_id, state, quantities=quantities)
        else:
            chosen = _resolve_choice(user_input, offered)
            if chosen:
                debug("recommendation_agent", f"user chose {chosen.get('item_name')!r}")
                return _reply_with_pick(
                    chosen, address_id, state=state,
                    awaiting_feedback=False, satisfied=True, last_recommendations=[chosen],
                )

        if not (is_category_offer or is_dish_multiselect_offer):
            reason = feedback.infer_reason(user_input)
            if not reason and recommendation_state.get("awaiting_reason") and _looks_like_rejection(user_input):
                reason = feedback.NOT_FOR_ME
            if reason:
                return _handle_rejection(reason, state, brief, recommendation_state, address_id)

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
                if is_dish_multiselect_offer:
                    return _handle_dish_multiselect([offered[0]], address_id, state)
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

        if _looks_like_rejection(user_input) and not (is_category_offer or is_dish_multiselect_offer):
            # A bare "no": one tap tells us what to fix (feature 2).
            debug("recommendation_agent", "user rejected the batch -- asking why")
            return _reply(
                feedback.REASON_QUESTION,
                **{**recommendation_state, "awaiting_reason": True, "options": feedback.REASON_OPTIONS},
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
