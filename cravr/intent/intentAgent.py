"""Intent Discovery agent: turns a food-ordering request into structured
intent. Owns the intent-clarification `interrupt()` loop -- extracts, derives, then
pauses once per still-missing field (category, then a specific dish/cuisine).
"""

from __future__ import annotations

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

from ..options import OTHER_VALUE, resolve_option_choice, suggest_options
from .prompt import INTENT_DISCOVERY_PROMPT



def _is_no_preference(answer: str) -> bool:
    normalized = (answer or "").strip().lower().rstrip(".!")
    return normalized in NO_PREFERENCE_ANSWERS


def _mentions_dietary_word(text: str, phrases: tuple[str, ...]) -> bool:
    """Whole-word/phrase match, so "veg" doesn't fire inside "vegetable"."""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) for phrase in phrases)


def _derive_dietary_preference(text: str) -> str | None:
    """Read an explicit veg/non-veg signal from free text -- a stated
    preference always wins over a saved default, even a one-off "actually,
    show me non-veg".

    Scanned as one list ranked longest-phrase-first, not tuple-by-tuple:
    "non-veg" contains "veg" and "no meat" (VEG) contains "meat" (NON_VEG),
    so checking either tuple whole first misreads the other's phrase. The
    longest match always wins regardless of which tuple it's in.
    """
    candidates = [(phrase, "Non-Veg") for phrase in NON_VEG_ANSWERS]
    candidates += [(phrase, "Veg") for phrase in VEG_ANSWERS]
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)
    for phrase, verdict in candidates:
        if _mentions_dietary_word(text, (phrase,)):
            return verdict
    return None


def extract_tastes(text: str) -> list[str]:
    """Map free wording onto the menu's five taste tags."""
    remaining = f" {text or ''} "
    found: list[str] = []

    # Consume the negation so "not spicy" can't also register as spicy.
    remaining, negations = MILD_PATTERN.subn(" ", remaining)
    if negations or any(word in remaining for word in MILD_WORDS):
        found.append("mild")

    for taste, cues in TASTE_CUES.items():
        if any(re.search(rf"(?<!\w){re.escape(cue)}", remaining) for cue in cues):
            if taste not in found:
                found.append(taste)

    return found


def _derive_flavor(profile: dict) -> str | None:
    """Normalize whatever the user said into one real menu tag. Reads
    craving *and* cuisine text -- "spicy" often arrives attached to a
    cuisine ("spicy Indian")."""
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
    """True when the user has told us anything to steer a menu with. A
    taste counts -- "something spicy" IS a craving; requiring a named dish
    here used to loop back and ask for one anyway."""
    if "craving_or_cuisine" in (profile.get("no_preference") or []):
        return True
    return bool(
        profile.get("craving") or profile.get("cuisine") or profile.get("flavor")
    )


def _derive(profile: dict, now: datetime | None = None) -> dict:
    """Fill every field the code can work out for itself. Runs after each
    extraction and each answer, so a fresh craving produces its flavor
    immediately rather than next turn."""
    derived = dict(profile)

    # The clock always knows this one; it is never worth asking.
    derived["meal_time"] = current_meal_window(now).name
    if not derived.get("meal_intent"):
        derived["meal_intent"] = current_meal_window(now).default_intent

    flavor = _derive_flavor(derived)
    if flavor:
        derived["flavor"] = flavor

    # Recomputed every pass unless stated -- a plain "if not hunger_level"
    # check isn't equivalent: the first derive (before meal_intent is known)
    # would write the DEFAULT_HUNGER placeholder and then block the real
    # value once "dinner" is answered a moment later.
    if not derived.get("hunger_level_stated"):
        derived["hunger_level"] = HUNGER_BY_INTENT.get(
            str(derived.get("meal_intent") or "").strip().lower(), DEFAULT_HUNGER
        )

    return derived


def _missing_fields(profile: dict) -> list[str]:
    """Which required fields are still unfilled, in the order they're asked.
    Category comes first, but only when there's no flavour signal at all yet --
    someone who already named a dish/cuisine skips the funnel entirely."""
    missing: list[str] = []
    no_pref = profile.get("no_preference") or []
    if not _has_flavour_signal(profile) and not profile.get("craving_tag") and "category" not in no_pref:
        missing.append("category")
    if not _has_flavour_signal(profile):
        missing.append("craving_or_cuisine")
    return missing


# Category funnel: asked first when nothing at all is known yet. Loosely mirrors
# Swiggy's own EATRIGHT/STORE_99 storefront collections (see food_mcp._infer_collection)
# so a pick here can also narrow the real search, not just phrase the next question.
_CATEGORY_OPTION_FALLBACK = [
    "Healthy", "Comfort food", "Something sweet", "High protein", "Budget-friendly",
]

# Safety net when the LLM can't be reached -- tappable choices instead of a bare prompt.
_TYPE_OPTION_FALLBACKS_BY_CATEGORY = {
    "comfort-food": ["Butter chicken", "Mac and cheese", "Biryani", "Pizza", "Noodles"],
    "lightweight": ["Salad", "Soup", "Grilled chicken", "Poke bowl", "Sushi"],
    "high-protein": ["Grilled chicken", "Paneer bowl", "Egg curry", "Sushi", "Protein bowl"],
    "low-calorie": ["Salad", "Soup", "Grilled fish", "Sushi", "Steamed veggies"],
    "dessert": ["Cake", "Ice cream", "Brownie", "Pastry", "Gulab jamun"],
}
_DEFAULT_CRAVING_OPTION_FALLBACK = [
    "A dish in mind", "A cuisine", "Spicy or sweet",
    "Healthy option", "Comfort food",
]


def _question_for(
    field: str, now: datetime | None = None, profile: dict | None = None
) -> tuple[str, list[dict]]:
    """The question to ask for a field, plus any options to offer."""
    if field == "category":
        question = "What are you in the mood for?"
        options = suggest_options(
            question,
            context={
                "meal_time": (profile or {}).get("meal_time"),
                "hunger_level": (profile or {}).get("hunger_level"),
            },
            fallback=_CATEGORY_OPTION_FALLBACK,
        )
        return question, options
    if field == "craving_or_cuisine":
        category = (profile or {}).get("craving_tag")
        category_label = category.replace("-", " ") if category else None
        question = (
            f"Great — {category_label}! What sounds good — a specific dish, a cuisine, "
            "or something else?"
            if category_label else
            "What are you in the mood for — a dish, a cuisine, or just a flavour "
            "like spicy or sweet?"
        )
        options = suggest_options(
            question,
            context={
                "category": category,
                "meal_time": (profile or {}).get("meal_time"),
                "hunger_level": (profile or {}).get("hunger_level"),
            },
            fallback=_TYPE_OPTION_FALLBACKS_BY_CATEGORY.get(category, _DEFAULT_CRAVING_OPTION_FALLBACK),
        )
        return question, options
    question = f"Could you tell me a bit more about {field.replace('_', ' ')}?"
    return question, suggest_options(question, context=profile or {}, fallback=[])


def _apply_answer(profile: dict, field: str, answer: str, now: datetime | None = None) -> dict:
    """Write a reply straight into the field it was asked about -- no
    round-trip through the extraction model, which routinely drops a bare
    "Indian" or "2" that's unambiguous here."""
    updated = dict(profile)
    answer = (answer or "").strip()
    if not answer:
        return updated

    if _is_no_preference(answer):
        # Record as *resolved*, not a literal "anything" value -- else the loop re-asks forever.
        resolved = list(updated.get("no_preference") or [])
        if field not in resolved:
            resolved.append(field)
        updated["no_preference"] = resolved
        return updated

    if field == "category":
        normalized = answer.lower()
        if normalized.isdigit() or len(normalized) < 2:
            return updated
        # Best-effort onto a known tag (e.g. "Healthy" -> "low-calorie" via its cue
        # words); an unrecognised label is still kept raw -- it still scopes the next
        # question, just doesn't also narrow the search filters.
        tag = next(
            (t for t, cues in CRAVING_TAG_CUES.items() if any(cue in normalized for cue in cues)),
            None,
        )
        updated["craving_tag"] = tag or answer
        return updated

    if field == "craving_or_cuisine":
        normalized = answer.lower()
        # No numbered options here, so a bare "1" is a stray keystroke, never a dish.
        if normalized.isdigit() or len(normalized) < 2:
            return updated
        matched_cuisine = next(
            (word for word in KNOWN_CUISINE_WORDS if word in normalized), None
        )
        if matched_cuisine:
            updated["cuisine"] = answer if len(answer) <= 30 else matched_cuisine
        else:
            updated["craving"] = answer
        return updated

    updated[field] = answer
    return updated


def intent_discovery_node(state: CravrState) -> dict:
    """Extract, derive, then interrupt until the required intent is
    complete. Safe to loop `interrupt()` here: LangGraph replays
    already-answered interrupts from stored resume values on every re-run,
    and `profile` is rebuilt deterministically from those replayed answers
    via `_apply_answer`, not from the checkpoint -- so every round survives.
    The extraction call re-runs on each resume too, harmlessly, since
    `merge_profile` only adds non-empty values.

    cart_agent (see cart/add_flow.py's _ask_for_addon_choices) is the only
    other node that calls `interrupt()` -- decision_agent_node never does
    (see `DecisionAgentNeverAsksTests`).
    """
    now = now_ist()

    # The CLI starts with no user message on purpose (offers a meal choice
    # first) -- never extract from an empty conversation, some models invent a craving.
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
        merged["hunger_level_stated"] = True  # sticky: outlasts the meal_intent default

    # Checked deterministically against the raw message (see
    # _derive_dietary_preference), not left to extraction. Overwrites rather
    # than unions -- a fresh "make it non-veg" must replace "Veg" outright,
    # not sit alongside it.
    dietary_signal = _derive_dietary_preference(latest_user_text(state.get("messages")))
    if dietary_signal:
        merged["dietary_preferences"] = [dietary_signal]

    profile = _derive(merged, now)
    debug("intent_agent", f"merged+derived profile={profile}")

    transcript: list = []
    asked: list[str] = []

    for round_number in range(1, MAX_INTENT_CLARIFICATION_ROUNDS + 1):
        pending = [f for f in _missing_fields(profile) if f not in asked]
        if not pending:
            break

        field = pending[0]
        question, options = _question_for(field, now, profile)
        debug("intent_agent", f"asking (round {round_number}) about {field!r}: {question!r}")

        raw_answer = interrupt({
            "question": question,
            "options": options,
            "agent": "intent_agent",
            "field": field,
        })
        debug("intent_agent", f"got answer for {field!r}: {raw_answer!r}")

        asked.append(field)  # never re-ask a field, even if the reply couldn't be used
        # Both sides go into `messages` -- a bare "2" is meaningless without
        # the question that prompted it; recorded as typed, before option resolution.
        transcript.append(AIMessage(content=question, name="intent_agent"))
        transcript.append(HumanMessage(content=str(raw_answer)))

        # A digit or the option's label/value resolves as if typed out --
        # lets one keystroke answer; anything unanticipated passes through as free text.
        answer = resolve_option_choice(str(raw_answer), options)
        if answer == OTHER_VALUE:
            # "type your own" -- ask again with no options for an unambiguous free-text reply.
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
    profile["clarification_questions"] = [
        _question_for(field, now, profile)[0] for field in still_missing
    ]

    updates: dict = {"intent_profile": profile}
    if transcript:
        updates["messages"] = transcript
    return updates
