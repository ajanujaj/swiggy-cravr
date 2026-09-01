"""Recommendation agent: uses the MCP recommendation tool and a human-in-the-loop feedback cycle."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..debug_utils import debug
from ..intent import KNOWN_CUISINE_WORDS, NON_VEG_ANSWERS, VEG_ANSWERS, extract_tastes
from ..llm import get_llm
from ..profile_utils import latest_user_text
from ..state import CravrState


DEFAULT_MCP_URL = os.getenv("CRAVR_MCP_URL", "http://127.0.0.1:8000/mcp")

# The only values the menu's `dietary_type` column actually holds. The MCP tool
# compares with `==`, so anything outside this set silently matches zero rows --
# which is exactly what "vegetarian" (vs. the stored "Veg") was doing.
DIETARY_TYPE_BY_PREFERENCE = {
    "veg": "Veg",
    "vegetarian": "Veg",
    "pure veg": "Veg",
    "veggie": "Veg",
    # No pure-vegan dietary_type exists in the menu; "Veg" plus the `vegan`
    # tag is the closest faithful filter.
    "vegan": "Veg",
    "non-veg": "Non-Veg",
    "non veg": "Non-Veg",
    "nonveg": "Non-Veg",
    "non-vegetarian": "Non-Veg",
    "non vegetarian": "Non-Veg",
    "eggetarian": "Non-Veg",
    "meat": "Non-Veg",
}

# The closed tag vocabulary in menu.csv. Anything outside it can only ever
# match zero rows, so it must not be sent as a `tags` filter.
KNOWN_TAGS = {
    "breakfast", "comfort-food", "dessert", "gluten-free", "high-fiber",
    "high-protein", "lightweight", "low-calorie", "low-carb", "mild",
    "one-pot-meal", "organic", "savory", "spicy", "sugar-free-option",
    "sweet", "tangy", "vegan", "vegan-option", "vegetarian",
}

# What the user is actually after, matched against menu.csv's `course` column.
# Ordering matters: the first hit wins, so beverage/dessert cues are checked
# before the generic meal default.
BEVERAGE_CUES = ("coffee", "tea", "chai", "drink", "beverage", "juice", "smoothie",
                 "latte", "cappuccino", "espresso", "shake", "lassi")
DESSERT_CUES = ("dessert", "sweet dish", "pudding", "cake", "brownie", "ice cream",
                "sweet tooth", "something sweet", "pastry")
SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "small plate",
             "nibble", "finger food")

# Anything the user could plausibly eat as a meal. Sides are included as a
# fallback rather than a first choice -- see `_course_preference`.
MEAL_COURSES = "main,side"

# How many dishes to put in front of the user at once.
DISPLAY_COUNT = 3

ACCEPTANCE_TOKENS = ("yes", "yep", "yeah", "sure", "ok", "okay", "perfect",
                     "great", "sounds good", "looks good", "love it",
                     "go ahead", "that works", "book it", "accept")
REJECTION_TOKENS = ("no", "nope", "nah", "not that", "different", "another",
                    "change", "new one", "something else", "skip", "reject")
# "show me more" is its own intent, distinct from rejection: the user isn't
# unhappy with what we offered, they just want to see the rest. Both serve the
# next batch, but only this one is honest about running out.
MORE_TOKENS = ("more", "other", "others", "else", "next", "alternatives",
               "alternative", "rest", "again")


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    return latest_user_text(messages)


def _mentions(user_input: str, tokens: tuple[str, ...]) -> bool:
    """Whole-word/phrase match.

    Plain substring matching made "I want noodles" read as a rejection (it
    contains "no") and "not good" read as an acceptance (it contains "good").
    """
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
    """Offer up to three numbered options rather than a single take-it-or-leave-it.

    One suggestion forces a yes/no on a stranger's taste; three lets the user
    choose. If fewer than three matched, we show what there is instead of
    padding with dishes that don't fit.
    """
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

# Numbered so the single-match confirmation costs one keystroke too, same as
# the multi-dish list above -- "Want it, or shall I widen the search?" used
# to be a plain yes/no the user had to type out in full.
ONLY_MATCH_PROMPT = "    1. Yes, I'll take it\n    2. Widen the search"


def _numbered(dishes: list[dict[str, Any]]) -> str:
    return "\n".join(f"  {i}. {_describe(d)}" for i, d in enumerate(dishes, start=1))


# Ordinals are matched before cardinals, and that ordering is load-bearing:
# in "the third one", the word "one" is a filler noun, not a choice. Checking
# cardinals first would silently select option 1 for "no, the second one".
ORDINAL_WORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}
CARDINAL_WORDS = {"one": 1, "two": 2, "three": 3}


def _resolve_choice(user_input: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Work out which of the offered dishes the user meant.

    Handles "2", "the second one", "#3" and "the biryani". Checked before the
    generic yes/no parsing, so "no, the second one" selects rather than rejects.
    """
    normalized = (user_input or "").strip().lower()
    if not normalized or not options:
        return None

    for word, position in ORDINAL_WORDS.items():
        if position <= len(options) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized):
            return options[position - 1]

    # Bare cardinals only count as a pick when they ARE the message ("two") or
    # follow an explicit marker ("number two"). Embedded in a sentence they're
    # almost always filler -- "the third one", "the thai one".
    for word, position in CARDINAL_WORDS.items():
        if position > len(options):
            continue
        if normalized.strip(" .!") == word or re.search(
            rf"(?:number|option|#)\s*{re.escape(word)}(?!\w)", normalized
        ):
            return options[position - 1]

    match = re.search(r"(?<!\d)([1-9])(?!\d)", normalized)
    if match and int(match.group(1)) <= len(options):
        return options[int(match.group(1)) - 1]

    # Name match: require a reasonably distinctive word so that a stray "rice"
    # or "veg" doesn't silently pick a dish the user didn't mean.
    for dish in options:
        words = [w for w in re.findall(r"\w+", str(dish.get("item_name") or "").lower()) if len(w) >= 5]
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", normalized) for w in words):
            return dish

    return None


def _course_preference(intent_profile: dict[str, Any]) -> str:
    """Infer which `course` the user is asking for.

    Defaults to "main" rather than leaving the field open. Without it the tool
    happily answers "I want something spicy" with a ₹70 Filter Coffee or a
    portion of Garlic Naan -- technically matching, useless as dinner.
    """
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
    # A sweet tooth expressed only as a taste preference is a weaker signal
    # than the word "dessert", but still enough to include sweet courses.
    if "sweet" in [str(t).lower() for t in intent_profile.get("taste_preference") or []]:
        return "dessert,main"
    if any(cue in haystack for cue in SIDE_CUES):
        return "side"
    return "main"


def _build_recommendation_filters(intent_profile: dict[str, Any]) -> dict[str, Any]:
    """Build the MCP `recommendation` tool's argument dict from intent_profile.

    Kept separate from `fetch_recommendations` so the filter-building logic
    can be unit tested without a live MCP server. Only pass keys the tool
    actually accepts (cuisine, restaurant_name, tags, dietary_type,
    meal_type, max_price_inr) -- forwarding anything else (e.g. a free-text
    `summary`) makes the whole call fail argument validation.

    Two rules matter more than they look:

    1. `cuisine` NEVER goes into `tags`. Cuisine lives in its own column and
       is already filtered on; no menu row carries "Indian" as a *tag*, so
       adding it there ANDed the result set down to zero every single time.
    2. `dietary_type` must be translated into the menu's own vocabulary
       ("Veg"/"Non-Veg"). The tool compares exactly, so the profile's
       "vegetarian" never matched anything.
    """
    filters: dict[str, Any] = {}

    tag_terms: list[str] = []
    for term in [intent_profile.get("craving"), *(intent_profile.get("taste_preference") or [])]:
        if not term:
            continue
        normalized = str(term).strip().lower()
        # Only real tags survive; a free-text craving like "biryani" isn't a
        # tag and would zero the result set if forwarded.
        if normalized in KNOWN_TAGS and normalized not in tag_terms:
            tag_terms.append(normalized)
    if tag_terms:
        # Comma-separated tags are AND-matched by the MCP recommendation tool.
        filters["tags"] = ",".join(tag_terms)

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
    """Boil order history down to the two things that inform a recommendation.

    "recent_orders=28; last_restaurant=Punjab Grill" said almost nothing --
    one arbitrary recent row. What the dish choice can actually use is which
    cuisines the user keeps coming back to and which items they re-order.
    """
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
    # Support the older MCP response shape while clients migrate to one
    # profile object per user.
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
    """Freeze a request into everything the recommender needs, and nothing more.

    Called once by decision_agent when it decides to recommend. Producing this
    in one place means the profile-to-menu-filter translation (dietary
    vocabulary, tag vocabulary, course inference) happens exactly once, rather
    than being re-derived -- or worse, re-parsed out of prose -- downstream.
    """
    effective_intent = _apply_saved_preferences(
        _coerce_profile(intent_profile), preference_profile
    )
    filters = _build_recommendation_filters(effective_intent)
    history = _summarise_history(_coerce_profile(history_profile))
    return {"filters": filters, "history": history, "summary": _brief_summary(filters, history)}


def _filter_variants(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Strict filters first, then progressively looser fallbacks.

    `dietary_type` and `max_price_inr` are never relaxed -- those are hard
    constraints (a vegetarian must not be shown chicken, and a budget is a
    budget). Cuisine goes before tags: taste is usually the thing the user
    actually said out loud ("something spicy"), whereas cuisine is often our
    own inference, so a spicy Thai dish beats a mild North Indian one when
    something has to give.
    """
    # `course` is widened (main -> main,side) rather than dropped: giving up
    # on it entirely is how a beverage or a dessert ends up answering "I want
    # to eat something spicy". Widening is tried before sacrificing the
    # user's stated taste, since a spicy side beats a mild main.
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
    """Order results by how well they honour the *original* request.

    This has to be preference-aware, not rating-only. Once a filter has been
    relaxed the MCP tool happily returns everything that survives the hard
    constraints, and sorting those purely by rating recommended a ₹70 Filter
    Coffee to someone who asked for something spicy -- it simply had the best
    rating in the set. Scoring against the dropped preferences puts the spicy
    dishes back on top while still letting the fallback do its job.
    """
    wanted_tags = {t.strip().lower() for t in desired_tags if t}
    cuisine = (desired_cuisine or "").strip().lower()
    # The first listed course is the one actually wanted; the rest are the
    # widened fallbacks, so they should still rank below it.
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

# context_profile: dict[str, Any], context_profile is currently disabled, so the recommendation tool is only driven by intent_profile and history_profile.
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


def _call_recommendation_tool(filters: dict[str, Any]) -> list[dict[str, Any]]:
    async def _call_tool() -> Any:
        from fastmcp import Client

        async with Client(DEFAULT_MCP_URL) as client:
            return await client.call_tool(name="recommendation", arguments=filters)

    try:
        result = asyncio.run(_call_tool())
    except Exception as exc:
        # This used to be a bare `return []`, which made "the MCP server isn't
        # running" indistinguishable from "no dish matched". Both surfaced to
        # the user as the same shrug.
        debug("recommendation_agent", f"MCP call FAILED ({type(exc).__name__}: {exc}) at {DEFAULT_MCP_URL}")
        return []

    return _unwrap_tool_result(result)


def fetch_recommendations(brief: dict[str, Any]) -> list[dict[str, Any]]:
    """Call the MCP recommendation tool, relaxing preference filters until something matches.

    Takes only the frozen brief from decision_agent -- no profiles, no
    re-derivation.
    """
    brief = _coerce_profile(brief)
    filters = dict(brief.get("filters") or {})
    history = _coerce_profile(brief.get("history"))
    debug("recommendation_agent", f"brief summary={brief.get('summary')!r}")
    debug("recommendation_agent", f"strict filters={filters} history={history}")

    desired_tags = [t for t in str(filters.get("tags") or "").split(",") if t]
    desired_cuisine = filters.get("cuisine")
    desired_course = filters.get("course")

    for attempt, variant in enumerate(_filter_variants(filters), start=1):
        recommendations = _call_recommendation_tool(variant)
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
    brief: dict[str, Any], exclude_items: set[str] | None = None
) -> list[dict[str, Any]]:
    """Deliberately relax the search beyond the strict match.

    For when `fetch_recommendations` only turned up a single result and the
    user asked to see more rather than take it. Reuses the exact same
    relaxation cascade (`_filter_variants`), just starting one step past the
    strict variant that already ran -- so `dietary_type` and budget are still
    never relaxed, only cuisine/tags/course, same as any other fallback.
    """
    brief = _coerce_profile(brief)
    filters = dict(brief.get("filters") or {})
    history = _coerce_profile(brief.get("history"))
    exclude = {str(name).strip().lower() for name in (exclude_items or set())}

    desired_tags = [t for t in str(filters.get("tags") or "").split(",") if t]
    desired_cuisine = filters.get("cuisine")
    desired_course = filters.get("course")

    # [0] is the exact-match variant that produced the single result already
    # shown; skip straight to the first relaxed one.
    for attempt, variant in enumerate(_filter_variants(filters)[1:], start=1):
        recommendations = _call_recommendation_tool(variant)
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


# The single-match confirmation offers exactly two numbered choices (see
# `ONLY_MATCH_PROMPT`); "2" only means "widen the search" in that specific
# context -- against a real multi-dish list "2" still means "the second
# one", handled separately by `_resolve_choice`.
WIDEN_TOKENS = ("widen", "broaden", "expand", "wider")


def _looks_like_decline_only_offer(user_input: str) -> bool:
    normalized = (user_input or "").strip().lower().rstrip(".!")
    if normalized == "2":
        return True
    return _looks_like_rejection(user_input) or _mentions(user_input, WIDEN_TOKENS)


def is_pure_feedback(user_input: str, recommendation_state: dict[str, Any] | None) -> bool:
    """True when a message only reacts to the dishes we just offered.

    Used by the graph to skip intent/history/decision entirely for replies
    like "3" or "yes". Re-running the whole pipeline for those isn't just
    wasted LLM and MCP calls -- it feeds "3" to the extraction model, and a
    model that reads that as budget_inr=3 would set max_price_inr=3 and
    silently zero out every subsequent search.

    Deliberately conservative: anything carrying a new constraint ("no, make
    it non-veg", "something under 200") goes the long way round so the profile
    actually gets updated.
    """
    state = recommendation_state or {}
    offered = state.get("last_recommendations") or []
    if not offered or not state.get("awaiting_feedback") or state.get("satisfied"):
        return False

    # An explicit pick is unambiguous even if it collides with other vocabulary.
    if _resolve_choice(user_input, offered):
        return True
    # Same reasoning for the single-match confirmation's "2": there's no
    # dish #2 to resolve above, but a bare "2" is just as unambiguous a click
    # as a real positional pick, so it skips the new-constraints gate too --
    # otherwise this one reply would take the full intent/context/decision
    # round-trip just to land back on the widen branch in
    # `recommendation_agent_node`. A typed-out "widen the search" isn't
    # this unambiguous (it could be riding along with a real new
    # constraint), so it still goes the long way round via that gate below.
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


def _serve_next_batch(
    lead_in: str,
    offered: list[dict[str, Any]],
    alternates: list[dict[str, Any]],
    filters: dict[str, Any],
) -> dict:
    """Show the next slice of an already-fetched result set.

    The full ranked list is kept in `alternates` precisely so "show me more"
    doesn't mean re-running the same query and re-printing the same three
    dishes -- which is exactly what "recommend more" used to do.
    """
    if alternates:
        batch, rest = alternates[:DISPLAY_COUNT], alternates[DISPLAY_COUNT:]
        debug("recommendation_agent", f"serving {len(batch)} more, {len(rest)} still held back")
        return _reply(
            f"{lead_in}\n{_numbered(batch)}\n{PICK_PROMPT}",
            awaiting_feedback=True, satisfied=False,
            last_recommendations=batch, alternates=rest, filters=filters,
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

    Reads the frozen `recommendation_brief` produced by decision_agent and
    nothing else. The fallback below exists only so the node stays callable in
    isolation (tests, a direct invoke); in the graph the brief is always there.
    """
    brief = _coerce_profile(state.get("recommendation_brief"))
    if not brief:
        debug("recommendation_agent", "no brief in state — deriving one from the profiles")
        brief = build_recommendation_brief(
            _coerce_profile(state.get("intent_profile")),
            _coerce_profile(state.get("history_profile")),
        )

    brief_filters = dict(brief.get("filters") or {})
    recommendation_state = dict(state.get("recommendation_state") or {})
    awaiting_feedback = bool(recommendation_state.get("awaiting_feedback"))
    offered = recommendation_state.get("last_recommendations") or []
    alternates = recommendation_state.get("alternates") or []
    satisfied = bool(recommendation_state.get("satisfied"))

    user_input = _extract_latest_user_input(state.get("messages") or [])
    debug("recommendation_agent", f"user_input={user_input!r} awaiting_feedback={awaiting_feedback}")

    if awaiting_feedback and not satisfied:
        # Check for an explicit pick first: "no, the second one" is a choice,
        # not a rejection.
        chosen = _resolve_choice(user_input, offered)
        if chosen:
            debug("recommendation_agent", f"user chose {chosen.get('item_name')!r}")
            return _reply(
                f"Great — {_describe(chosen)} it is.",
                awaiting_feedback=False, satisfied=True, last_recommendations=[chosen],
            )

        # The single-match confirmation's own "2" (see `ONLY_MATCH_PROMPT`),
        # or a typed-out decline/widen, actually broadens the search rather
        # than just asking the user to spell out a change -- checked ahead of
        # the generic rejection branch below, which only offers to change,
        # never to search again itself.
        if len(offered) == 1 and _looks_like_decline_only_offer(user_input):
            debug("recommendation_agent", "user declined the only match — widening the search")
            already_shown = {str(d.get("item_name") or "") for d in offered}
            widened = widen_recommendations(brief, exclude_items=already_shown)
            if widened:
                batch, rest = widened[:DISPLAY_COUNT], widened[DISPLAY_COUNT:]
                return _reply(
                    f"Here's what turns up with a wider search:\n{_numbered(batch)}\n{PICK_PROMPT}",
                    awaiting_feedback=True, satisfied=False,
                    last_recommendations=batch, alternates=rest, filters=brief_filters,
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
                return _reply(
                    f"Great — {_describe(offered[0])} it is.",
                    awaiting_feedback=False, satisfied=True, last_recommendations=offered,
                )
            # "yes" against a list of three is ambiguous; don't guess.
            return _reply(
                f"Which one would you like — {' or '.join(str(i) for i in range(1, len(offered) + 1))}?",
                awaiting_feedback=True, satisfied=False,
                last_recommendations=offered, alternates=alternates, filters=brief_filters,
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
            # Not a recognised reaction, but the request hasn't changed either --
            # so re-running the search would return the identical list and we'd
            # print the same three dishes back. Show what we're holding instead.
            debug("recommendation_agent", "unrecognised reply, filters unchanged — treating as 'more'")
            return _serve_next_batch("Here are a few more:", offered, alternates, brief_filters)

    ranked = fetch_recommendations(brief)
    if not ranked:
        return _reply(
            "I’m not finding a strong match right now. Tell me your cuisine, budget, "
            "or dietary preference and I’ll refine it.",
            awaiting_feedback=False, satisfied=False, last_recommendations=[],
        )

    batch, rest = ranked[:DISPLAY_COUNT], ranked[DISPLAY_COUNT:]
    debug("recommendation_agent", f"offering {len(batch)} of {len(ranked)} match(es)")
    return _reply(
        _build_recommendation_text(batch),
        awaiting_feedback=True, satisfied=False,
        last_recommendations=batch, alternates=rest, filters=brief_filters,
    )
