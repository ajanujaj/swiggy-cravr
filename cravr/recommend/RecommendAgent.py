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

    for dish in options:
        words = [w for w in re.findall(r"\w+", str(dish.get("item_name") or "").lower()) if len(w) >= 5]
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", normalized) for w in words):
            return dish

    return None


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
) -> dict:
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
        chosen = _resolve_choice(user_input, offered)
        if chosen:
            debug("recommendation_agent", f"user chose {chosen.get('item_name')!r}")
            return _reply_with_pick(
                chosen, address_id, state=state,
                awaiting_feedback=False, satisfied=True, last_recommendations=[chosen],
            )

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
