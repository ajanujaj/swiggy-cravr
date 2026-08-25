"""Intent Discovery agent: turns a food-ordering request into structured intent.

This node now owns *all* clarification. It extracts what it can from the
conversation, derives what the clock and its own vocabulary can supply, and
then pauses the graph with `interrupt()` -- once per still-missing field --
until its required state is complete. `decision_agent` no longer asks
questions; it only judges.
"""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from ..debug_utils import debug
from ..llm import get_llm
from ..profile_utils import merge_profile
from ..schemas import IntentExtraction
from ..state import CravrState
from .meal_clock import (
    current_meal_window,
    meal_intent_options,
    meal_intent_question,
    resolve_meal_intent,
)


INTENT_DISCOVERY_PROMPT = """
You are Cravr's Intent Discovery agent. Analyze the user's food-ordering message(s).
You are not a conversational assistant: do not recommend food and do not ask a
question. Extract only facts explicitly stated or safely implied by the message.
Use null/empty values for anything not mentioned. Never fabricate values.

Field notes:
- meal_intent: what the user wants this order to BE -- breakfast, lunch, dinner,
  snack, quick_bite, quick_lunch, late_night_craving, celebration. Only set it
  when the user says or clearly implies it; do not infer it from the clock.
- craving: the dish or food the user named, in their own words ("biryani",
  "something with paneer"). Free text.
- flavor: normalize the craving/flavour onto exactly one of: spicy, mild,
  sweet, savory, tangy, comfort-food, lightweight, high-protein, low-calorie,
  dessert. Leave null if none of them fit.
- cuisine: e.g. North Indian, Chinese, Italian, Thai.
- hunger_level: low, medium or high -- only when stated ("starving", "not very
  hungry").

Users describe flavour in many ways; match intent, not the literal word:
- spicy: "spicy", "hot", "extra hot", "fiery", "with chilli", "make it hot"
- mild: "mild", "not spicy", "less spicy", "no chilli", "nothing too hot"
- sweet: "sweet", "sweet tooth", "something sugary", "sweet treat"
- savory: "savory", "salty", "umami", "hearty"
- tangy: "tangy", "tart", "sour", "citrusy", "zesty"
Never guess a flavour the user did not express.
""".strip()


MAX_INTENT_CLARIFICATION_ROUNDS = 3

KNOWN_CUISINE_WORDS = (
    "north indian", "south indian", "indian", "chinese", "italian", "mexican",
    "japanese", "thai", "continental", "punjabi", "mughlai", "asian",
)

NO_PREFERENCE_ANSWERS = (
    "anything", "any", "no preference", "nothing", "none", "doesn't matter",
    "does not matter", "dont matter", "don't matter", "you decide", "you choose",
    "surprise me", "whatever", "up to you", "no idea", "not sure", "idk",
)

TASTE_CUES = {
    "spicy": ("spicy", "hot", "fiery", "chilli", "chili", "masala"),
    "sweet": ("sweet", "sugary", "sweet tooth"),
    "savory": ("savory", "savoury", "salty", "umami", "hearty"),
    "tangy": ("tangy", "tart", "sour", "citrusy", "zesty"),
}

MILD_PATTERN = re.compile(
    r"\b(?:not|no|nothing|less|avoid|without)\s+(?:too\s+|very\s+)?(?:spicy|hot|chill?i)\b"
)
MILD_WORDS = ("mild", "bland")

VEG_ANSWERS = ("vegetarian", "veg", "pure veg", "no meat", "veggie")
NON_VEG_ANSWERS = ("non-veg", "non veg", "nonveg", "non-vegetarian", "non vegetarian",
                   "meat", "chicken", "mutton", "fish", "egg", "eggetarian")

CRAVING_TAG_CUES = {
    "comfort-food": ("comfort", "homely", "home food", "cosy", "cozy"),
    "lightweight": ("light", "lightweight"),
    "high-protein": ("protein", "high protein"),
    "low-calorie": ("low calorie", "low-cal", "diet", "healthy"),
    "dessert": ("dessert", "pudding", "cake", "ice cream", "brownie", "pastry"),
}

HUNGER_BY_INTENT = {
    "breakfast": "medium",
    "lunch": "high",
    "quick_lunch": "medium",
    "dinner": "high",
    "celebration": "high",
    "snack": "low",
    "quick_bite": "low",
    "late_night_craving": "low",
}
DEFAULT_HUNGER = "medium"

REQUIRED_FIELDS = ("meal_intent", "craving_or_cuisine")


def _is_no_preference(answer: str) -> bool:
    normalized = (answer or "").strip().lower().rstrip(".!")
    return normalized in NO_PREFERENCE_ANSWERS


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
    """Normalize whatever the user said into one real menu tag.

    Reads craving *and* cuisine text, because "spicy" often arrives attached to
    a cuisine ("spicy Indian") rather than as a standalone craving.
    """
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
    """True when the user has told us anything we can steer a menu with.

    A taste counts: "something spicy" IS a craving. Requiring a named dish here
    is what used to make Cravr answer "I want something spicy" with "do you
    have a specific craving in mind?".
    """
    if "craving_or_cuisine" in (profile.get("no_preference") or []):
        return True
    return bool(
        profile.get("craving") or profile.get("cuisine") or profile.get("flavor")
    )


def _derive(profile: dict, now: datetime | None = None) -> dict:
    """Fill every field the code can work out for itself.

    Runs after each extraction and after each answer, so a reply that supplies
    a craving immediately produces its flavor rather than waiting for the
    next turn's LLM pass.
    """
    derived = dict(profile)

    # The clock always knows this one; it is never worth asking.
    derived["meal_time"] = current_meal_window(now).name

    flavor = _derive_flavor(derived)
    if flavor:
        derived["flavor"] = flavor

    # Recomputed every pass unless the user actually said it. A plain
    # `if not derived.get("hunger_level")` looks equivalent and isn't: the
    # first derive (before meal_intent is known) writes the DEFAULT_HUNGER
    # placeholder, which then blocks the real value once the user answers
    # "dinner" a moment later. Everyone came out "medium".
    if not derived.get("hunger_level_stated"):
        derived["hunger_level"] = HUNGER_BY_INTENT.get(
            str(derived.get("meal_intent") or "").strip().lower(), DEFAULT_HUNGER
        )

    return derived


def _missing_fields(profile: dict) -> list[str]:
    """Which required fields are still unfilled, in the order they're asked."""
    missing: list[str] = []
    if not profile.get("meal_intent"):
        missing.append("meal_intent")
    if not _has_flavour_signal(profile):
        missing.append("craving_or_cuisine")
    return missing


def _question_for(field: str, now: datetime | None = None) -> tuple[str, list[dict]]:
    """The question to ask for a field, plus any options to offer."""
    if field == "meal_intent":
        return meal_intent_question(now), meal_intent_options(now)
    if field == "craving_or_cuisine":
        return (
            "What are you in the mood for — a dish, a cuisine, or just a flavour "
            "like spicy or sweet?",
            [],
        )
    return f"Could you tell me a bit more about {field.replace('_', ' ')}?", []


def _apply_answer(profile: dict, field: str, answer: str, now: datetime | None = None) -> dict:
    """Write a reply straight into the field it was asked about.

    We know exactly which gap we just asked about, so the answer should never
    have to survive a round-trip through the extraction model to land in the
    profile. A bare "Indian" or "2" is unambiguous here and routinely dropped
    by a small model asked to infer it.
    """
    updated = dict(profile)
    answer = (answer or "").strip()
    if not answer:
        return updated

    if _is_no_preference(answer):
        # Record it as *resolved* rather than storing "anything" as a value --
        # otherwise the loop re-asks forever and the filter matches nothing.
        resolved = list(updated.get("no_preference") or [])
        if field not in resolved:
            resolved.append(field)
        updated["no_preference"] = resolved
        if field == "meal_intent":
            updated["meal_intent"] = current_meal_window(now).default_intent
        return updated

    if field == "meal_intent":
        updated["meal_intent"] = (
            resolve_meal_intent(answer, now) or current_meal_window(now).default_intent
        )
        return updated

    if field == "craving_or_cuisine":
        normalized = answer.lower()
        # This question has no numbered options, so a bare "1" is a stray
        # keystroke or an answer to the *previous* question -- never a dish.
        # Storing it would put `craving="1"` into the menu filters.
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
    """Extract, derive, then interrupt until the required intent is complete.

    The `interrupt()` calls sit in a loop inside this one node, which is safe
    here for a specific reason. LangGraph re-executes a node from the top on
    every resume, replaying already-answered `interrupt()` calls from its
    stored resume values. That only loses information if the loop rebuilds its
    state from the *checkpoint*; this one rebuilds `profile` from those replayed
    answers via `_apply_answer`, deterministically, so every round survives the
    replay. The one non-deterministic step -- the extraction call -- re-runs on
    each resume, which is harmless because `merge_profile` only ever adds
    non-empty values, though it does cost an extra model call per question.
    """
    now = datetime.now()

    try:
        structured_llm = get_llm().with_structured_output(IntentExtraction)
        extraction = structured_llm.invoke(
            [SystemMessage(content=INTENT_DISCOVERY_PROMPT), *state["messages"]]
        )
        extracted = extraction.model_dump()
        debug("intent_agent", f"structured extraction={extracted}")
    except Exception as exc:
        debug("intent_agent", f"structured output call failed ({exc}) — treating as no new information")
        extracted = {}

    merged = merge_profile(state.get("intent_profile"), extracted)
    if extracted.get("hunger_level"):
        # Sticky, so a hunger level stated three turns ago keeps beating the
        # meal_intent default for the rest of the conversation.
        merged["hunger_level_stated"] = True

    profile = _derive(merged, now)
    debug("intent_agent", f"merged+derived profile={profile}")

    transcript: list = []
    asked: list[str] = []

    for round_number in range(1, MAX_INTENT_CLARIFICATION_ROUNDS + 1):
        pending = [f for f in _missing_fields(profile) if f not in asked]
        if not pending:
            break

        field = pending[0]
        question, options = _question_for(field, now)
        debug("intent_agent", f"asking (round {round_number}) about {field!r}: {question!r}")

        answer = interrupt({
            "question": question,
            "options": options,
            "agent": "intent_agent",
            "field": field,
        })
        debug("intent_agent", f"got answer for {field!r}: {answer!r}")

        # A field is never asked twice. If the reply couldn't be used we move
        # on rather than repeating ourselves until the budget runs out.
        asked.append(field)
        # Both sides of the exchange go into `messages`: a bare reply like
        # "2" or "Indian" is meaningless to any later reader without the
        # question that prompted it.
        transcript.append(AIMessage(content=question, name="intent_agent"))
        transcript.append(HumanMessage(content=str(answer)))

        profile = _derive(_apply_answer(profile, field, str(answer), now), now)
        debug("intent_agent", f"profile after {field!r}={profile}")

    still_missing = _missing_fields(profile)
    if still_missing:
        debug("intent_agent", f"proceeding with gaps remaining: {still_missing}")
    profile["important_missing_information"] = still_missing
    profile["clarification_questions"] = [_question_for(f, now)[0] for f in still_missing]

    updates: dict = {"intent_profile": profile}
    if transcript:
        updates["messages"] = transcript
    return updates
