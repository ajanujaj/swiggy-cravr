from __future__ import annotations

import os
import re
from datetime import datetime

from ..clock import now_ist

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from ..debug_utils import debug
from ..llm import get_llm
from ..profile_utils import latest_user_text, merge_profile
from ..schemas import IntentExtraction
from ..state import CravrState
from .meal_clock import current_meal_window
from .meal_utils import (
    CRAVING_TAG_CUES,
    HUNGER_BY_INTENT,
    KNOWN_CUISINE_WORDS,
    MILD_PATTERN,
    MILD_WORDS,
    NO_PREFERENCE_ANSWERS,
    TASTE_CUES,
    VEG_ANSWERS,
    NON_VEG_ANSWERS,
    DEFAULT_HUNGER,
    MAX_INTENT_CLARIFICATION_ROUNDS,
)

from ..options import OTHER_VALUE, clear_stable, resolve_option_choice, stable, suggest_options
from .prompt import INTENT_DISCOVERY_PROMPT


def _is_no_preference(answer: str) -> bool:
    normalized = (answer or "").strip().lower().rstrip(".!")
    return normalized in NO_PREFERENCE_ANSWERS


def _mentions_dietary_word(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) for phrase in phrases)


def _derive_dietary_preference(text: str) -> str | None:
    candidates = [(phrase, "Non-Veg") for phrase in NON_VEG_ANSWERS]
    candidates += [(phrase, "Veg") for phrase in VEG_ANSWERS]
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)
    for phrase, verdict in candidates:
        if _mentions_dietary_word(text, (phrase,)):
            return verdict
    return None


def extract_tastes(text: str) -> list[str]:
    remaining = f" {text or ''} "
    found: list[str] = []

    remaining, negations = MILD_PATTERN.subn(" ", remaining)
    if negations or any(word in remaining for word in MILD_WORDS):
        found.append("mild")

    for taste, cues in TASTE_CUES.items():
        if any(re.search(rf"(?<!\w){re.escape(cue)}", remaining) for cue in cues):
            if taste not in found:
                found.append(taste)

    return found


def _derive_flavor(profile: dict) -> str | None:
    haystack = " ".join(
        str(profile.get(key) or "") for key in ("craving", "cuisine", "meal_intent")
    ).lower()
    if not haystack.strip():
        return None

    tastes = extract_tastes(haystack)
    if tastes:
        return tastes[0]

    for tag, cues in CRAVING_TAG_CUES.items():
        if any(cue in haystack for cue in cues):
            return tag

    return None


def _has_flavour_signal(profile: dict) -> bool:
    if "craving_or_cuisine" in (profile.get("no_preference") or []):
        return True
    return bool(
        profile.get("craving") or profile.get("cuisine") or profile.get("flavor")
    )


def _derive(profile: dict, now: datetime | None = None) -> dict:
    derived = dict(profile)

    derived["meal_time"] = current_meal_window(now).name
    if not derived.get("meal_intent"):
        derived["meal_intent"] = current_meal_window(now).default_intent

    flavor = _derive_flavor(derived)
    if flavor:
        derived["flavor"] = flavor

    if not derived.get("hunger_level_stated"):
        derived["hunger_level"] = HUNGER_BY_INTENT.get(
            str(derived.get("meal_intent") or "").strip().lower(), DEFAULT_HUNGER
        )

    return derived


def _should_open_with_dishes(state: dict, profile: dict) -> bool:
    """True when the request says nothing about what to eat, and the user
    hasn't just passed on the opening cards ("Something else" means: ask me)."""
    if os.getenv("CRAVR_DISHES_FIRST", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if _has_flavour_signal(profile) or profile.get("craving_tag"):
        return False
    rec = state.get("recommendation_state") or {}
    declined = (
        rec.get("offer_kind") == "opening"
        and rec.get("awaiting_feedback")
        and not rec.get("satisfied")
    )
    return not declined


def _missing_fields(profile: dict) -> list[str]:
    missing: list[str] = []
    no_pref = profile.get("no_preference") or []
    if not _has_flavour_signal(profile) and not profile.get("craving_tag") and "category" not in no_pref:
        missing.append("category")
    if not _has_flavour_signal(profile):
        missing.append("craving_or_cuisine")
    return missing


_CATEGORY_OPTION_FALLBACK = [
    "Healthy", "Comfort food", "Something sweet", "High protein", "Budget-friendly",
]

# India-first: these show when the LLM is unreachable, so they must be dishes
# most Swiggy areas actually deliver.
_TYPE_OPTION_FALLBACKS_BY_CATEGORY = {
    "comfort-food": ["Butter chicken", "Biryani", "Rajma chawal", "Pizza", "Hakka noodles"],
    "lightweight": ["Salad", "Soup", "Idli", "Poha", "Grilled chicken"],
    "high-protein": ["Grilled chicken", "Paneer tikka", "Egg curry", "Chicken tikka", "Protein bowl"],
    "low-calorie": ["Salad", "Soup", "Grilled fish", "Idli", "Sprouts chaat"],
    "dessert": ["Gulab jamun", "Ice cream", "Brownie", "Cake", "Rasmalai"],
}
_DEFAULT_CRAVING_OPTION_FALLBACK = [
    "A dish in mind", "A cuisine", "Spicy or sweet",
    "Healthy option", "Comfort food",
]


def _user_grounding(state: dict | None) -> dict:
    """What we know about this user's real ordering, so suggested options fit
    them (a vada-pav-and-chai regular shouldn't be offered a charcuterie board)."""
    state = state or {}
    grounding: dict = {}
    area = (state.get("context_profile") or {}).get("delivery_area")
    if area:
        grounding["delivery_area"] = area
    orders = (state.get("history_profile") or {}).get("orders") or []
    recent = []
    for order in orders[:5]:
        names = [i.get("name") for i in (order.get("items") or []) if i.get("name")]
        if not names and order.get("item_name"):
            names = [order["item_name"]]
        if order.get("restaurant_name") or names:
            recent.append({"restaurant": order.get("restaurant_name"), "items": names[:4]})
    if recent:
        grounding["recent_orders"] = recent
        counts: dict[str, int] = {}
        for entry in recent:
            if entry["restaurant"]:
                counts[entry["restaurant"]] = counts.get(entry["restaurant"], 0) + 1
        grounding["favourite_restaurants"] = sorted(counts, key=counts.get, reverse=True)[:3]
    prefs = (state.get("preference_profile") or {}).get("preferences") or {}
    if isinstance(prefs, dict) and prefs.get("dietary_preference"):
        grounding["dietary_preference"] = prefs["dietary_preference"]
    return grounding


def _question_text(field: str, profile: dict | None = None) -> str:
    if field == "category":
        return "What are you in the mood for?"
    if field == "craving_or_cuisine":
        category = (profile or {}).get("craving_tag")
        category_label = category.replace("-", " ") if category else None
        if category_label:
            return (
                f"Great — {category_label}! What sounds good — a specific dish, a cuisine, "
                "or something else?"
            )
        return (
            "What are you in the mood for — a dish, a cuisine, or just a flavour "
            "like spicy or sweet?"
        )
    return f"Could you tell me a bit more about {field.replace('_', ' ')}?"


def _question_for(
    field: str, now: datetime | None = None, profile: dict | None = None,
    state: dict | None = None,
) -> tuple[str, list[dict]]:
    profile = profile or {}
    question = _question_text(field, profile)
    grounding = _user_grounding(state)
    if field == "category":
        options = suggest_options(
            question,
            context={
                "meal_time": profile.get("meal_time"),
                "hunger_level": profile.get("hunger_level"),
                **grounding,
            },
            fallback=_CATEGORY_OPTION_FALLBACK,
        )
        return question, options
    if field == "craving_or_cuisine":
        category = profile.get("craving_tag")
        options = suggest_options(
            question,
            context={
                "category": category,
                "meal_time": profile.get("meal_time"),
                "hunger_level": profile.get("hunger_level"),
                **grounding,
            },
            fallback=_TYPE_OPTION_FALLBACKS_BY_CATEGORY.get(category, _DEFAULT_CRAVING_OPTION_FALLBACK),
        )
        return question, options
    return question, suggest_options(question, context={**profile, **grounding}, fallback=[])


def _apply_answer(profile: dict, field: str, answer: str, now: datetime | None = None) -> dict:
    updated = dict(profile)
    answer = (answer or "").strip()
    if not answer:
        return updated

    if _is_no_preference(answer):
        resolved = list(updated.get("no_preference") or [])
        if field not in resolved:
            resolved.append(field)
        updated["no_preference"] = resolved
        return updated

    if field == "category":
        normalized = answer.lower()
        if normalized.isdigit() or len(normalized) < 2:
            return updated
        tag = next(
            (t for t, cues in CRAVING_TAG_CUES.items() if any(cue in normalized for cue in cues)),
            None,
        )
        updated["craving_tag"] = tag or answer
        return updated

    if field == "craving_or_cuisine":
        normalized = answer.lower()
        if normalized.isdigit() or len(normalized) < 2:
            return updated
        matched_cuisine = next(
            (word for word in KNOWN_CUISINE_WORDS if word in normalized), None
        )
        if matched_cuisine:
            updated["cuisine"] = answer if len(answer) <= 30 else matched_cuisine
            if not updated.get("craving") and updated.get("craving_tag") not in CRAVING_TAG_CUES and updated.get("craving_tag"):
                updated["craving"] = updated["craving_tag"]
        else:
            updated["craving"] = answer
        return updated

    updated[field] = answer
    return updated


def intent_discovery_node(state: CravrState) -> dict:
    now = now_ist()

    if state.get("messages"):
        try:
            structured_llm = get_llm().with_structured_output(IntentExtraction)
            llm_messages = [SystemMessage(content=INTENT_DISCOVERY_PROMPT), *state["messages"]]
            debug("intent_agent", f"LLM request messages={llm_messages!r}")
            extraction = structured_llm.invoke(llm_messages)
            extracted = extraction.model_dump()
            debug("intent_agent", f"LLM response structured extraction={extracted}")
        except Exception as exc:
            debug("intent_agent", f"structured output call failed ({exc}) — treating as no new information")
            extracted = {}
    else:
        debug("intent_agent", "LLM extraction skipped: no user message yet")
        extracted = {}

    merged = merge_profile(state.get("intent_profile"), extracted)
    if extracted.get("hunger_level"):
        merged["hunger_level_stated"] = True

    dietary_signal = _derive_dietary_preference(latest_user_text(state.get("messages")))
    if dietary_signal:
        merged["dietary_preferences"] = [dietary_signal]

    profile = _derive(merged, now)
    profile.pop("opening_picks", None)  # never carried over from an earlier turn
    debug("intent_agent", f"merged+derived profile={profile}")

    if _should_open_with_dishes(state, profile):
        # Dishes first: an undecided user can't answer "what are you in the mood
        # for?", but can react to three concrete cards. Only if none can be
        # built do we fall through to the questions below.
        from ..recommend.opening import build_opening_picks

        picks = build_opening_picks({**state, "intent_profile": profile}, now)
        if picks:
            debug("intent_agent", f"opening with {len(picks)} dish card(s) instead of questions")
            profile["opening_picks"] = picks
            profile["important_missing_information"] = []
            profile["clarification_questions"] = []
            return {"intent_profile": profile}

    transcript: list = []
    asked: list[str] = []
    # Unchanged across interrupt replays within one user turn; changes on the next turn.
    turn_marker = len(state.get("messages") or [])

    for round_number in range(1, MAX_INTENT_CLARIFICATION_ROUNDS + 1):
        pending = [f for f in _missing_fields(profile) if f not in asked]
        if not pending:
            break

        field = pending[0]
        # Same question + options on every replay of this node (see options.stable).
        question, options = stable(
            ("intent", turn_marker, round_number, field),
            lambda: _question_for(field, now, profile, state),
        )
        debug("intent_agent", f"asking (round {round_number}) about {field!r}: {question!r}")

        raw_answer = interrupt({
            "question": question,
            "options": options,
            "agent": "intent_agent",
            "field": field,
        })
        debug("intent_agent", f"got answer for {field!r}: {raw_answer!r}")

        asked.append(field)
        transcript.append(AIMessage(content=question, name="intent_agent"))
        transcript.append(HumanMessage(content=str(raw_answer)))

        answer = resolve_option_choice(str(raw_answer), options)
        if answer == OTHER_VALUE:
            follow_up = "Sure — go ahead and tell me in your own words."
            debug("intent_agent", f"'other' chosen for {field!r}; asking free-text follow-up")
            other_answer = interrupt({
                "question": follow_up,
                "options": [],
                "agent": "intent_agent",
                "field": field,
            })
            debug("intent_agent", f"got free-text answer for {field!r}: {other_answer!r}")
            transcript.append(AIMessage(content=follow_up, name="intent_agent"))
            transcript.append(HumanMessage(content=str(other_answer)))
            answer = str(other_answer)

        profile = _derive(_apply_answer(profile, field, str(answer), now), now)

        debug("intent_agent", f"profile after {field!r}={profile}")

    still_missing = _missing_fields(profile)
    if still_missing:
        debug("intent_agent", f"proceeding with gaps remaining: {still_missing}")
    profile["important_missing_information"] = still_missing
    # Text only -- building options here cost an LLM call per field for nothing.
    profile["clarification_questions"] = [_question_text(field, profile) for field in still_missing]
    clear_stable(("intent", turn_marker))

    updates: dict = {"intent_profile": profile}
    if transcript:
        updates["messages"] = transcript
    return updates
