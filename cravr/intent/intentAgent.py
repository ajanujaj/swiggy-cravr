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
from ..profile_utils import latest_user_text, merge_profile
from ..schemas import IntentExtraction
from ..state import CravrState
from .meal_clock import (
    current_meal_window,
    meal_intent_options,
    meal_intent_question,
    resolve_meal_intent,
)
from .meal_utils import (
    CRAVING_TAG_CUES,
    HUNGER_BY_INTENT,
    KNOWN_CUISINE_WORDS,
    MILD_PATTERN,
    MILD_WORDS,
    NO_PREFERENCE_ANSWERS,
    REQUIRED_FIELDS,
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
    """Read an explicit veg/non-veg signal straight out of free text.

    A saved profile default (e.g. "Veg") is only ever a fallback for when
    the request says nothing -- once the user states a preference in this
    conversation, that has to win, including a one-off "actually, show me
    non-veg" that contradicts their saved default.

    Checked as one ranked scan across both tuples, longest phrase first,
    rather than one whole tuple before the other: several phrases in one
    tuple contain a shorter phrase from the *other* tuple as a plain
    substring, and the word-boundary check in `_mentions_dietary_word` can't
    tell them apart on its own --
      - "non-veg" contains "veg": checking VEG_ANSWERS first would read
        "what non-veg options" as vegetarian.
      - "no meat" (a VEG_ANSWERS alias) contains "meat" (a NON_VEG_ANSWERS
        alias): checking NON_VEG_ANSWERS first would read "no meat please"
        as non-veg -- the opposite of what it says.
    Ranking every phrase by length and taking the first match resolves both:
    the longer, more specific phrase always wins over a shorter one it
    contains, whichever tuple it happens to be in.
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


# Static safety net for the craving_or_cuisine question: used whenever the
# LLM can't be reached (offline, model not pulled, malformed output) so the
# user still gets tappable choices instead of a bare "type anything" prompt.
# Kept intentionally close to the follow-up phrasing below, per meal_intent.
_CRAVING_OPTION_FALLBACKS = {
    "breakfast": ["Something light", "Healthy meal", "Filling breakfast"],
    "quick_bite": ["Something sweet", "Something spicy", "Something healthy"],
    "lunch": ["Full meal", "Healthy meal", "Indian cuisine"],
    "quick_lunch": ["Something light", "Something filling", "Easy to grab"],
    "snack": ["Something spicy", "Something sweet", "Something healthy"],
    "dinner": ["Full meal", "Healthy meal", "Indian cuisine"],
    "late_night_craving": ["Something sweet", "Something savoury", "Something spicy"],
}
_DEFAULT_CRAVING_OPTION_FALLBACK = ["A dish in mind", "A cuisine", "Spicy or sweet"]


def _question_for(
    field: str, now: datetime | None = None, profile: dict | None = None
) -> tuple[str, list[dict]]:
    """The question to ask for a field, plus any options to offer."""
    if field == "meal_intent":
        return meal_intent_question(now), meal_intent_options(now)
    if field == "craving_or_cuisine":
        meal_intent = (profile or {}).get("meal_intent")
        follow_ups = {
            "breakfast": (
                "Breakfast it is! What are you in the mood for — something light, healthy, "
                "or filling?"
            ),
            "quick_bite": (
                "Got it — something quick. Are you craving sweet, spicy, healthy, or indulgent?"
            ),
            "lunch": (
                "Great — lunch it is! What are you in the mood for today?"
            ),
            "quick_lunch": (
                "Got it — a quick lunch. Do you want something light, filling, or easy to grab?"
            ),
            "snack": (
                "Snack it is! Are you craving something spicy, sweet, crunchy, or healthy?"
            ),
            "dinner": (
                "Dinner it is! What kind of meal are you in the mood for?"
            ),
            "late_night_craving": (
                "Late-night craving detected 😄 What are you craving — sweet, savoury, spicy, or comforting?"
            ),
        }
        question = follow_ups.get(
            meal_intent,
            "What are you in the mood for — a dish, a cuisine, or just a flavour "
            "like spicy or sweet?",
        )
        options = suggest_options(
            question,
            context={
                "meal_intent": meal_intent,
                "meal_time": (profile or {}).get("meal_time"),
                "hunger_level": (profile or {}).get("hunger_level"),
            },
            fallback=_CRAVING_OPTION_FALLBACKS.get(meal_intent, _DEFAULT_CRAVING_OPTION_FALLBACK),
        )
        return question, options
    question = f"Could you tell me a bit more about {field.replace('_', ' ')}?"
    return question, suggest_options(question, context=profile or {}, fallback=[])


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

    # The CLI intentionally starts with no user message so it can offer a
    # meal choice first. Never ask the extraction model to infer preferences
    # from that empty conversation: some models invent a craving, which skips
    # the next question entirely.
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
        # Sticky, so a hunger level stated three turns ago keeps beating the
        # meal_intent default for the rest of the conversation.
        merged["hunger_level_stated"] = True

    # Checked deterministically, against the raw message, rather than left to
    # the extraction model above -- see `_derive_dietary_preference`. Only
    # overwrites when *this* message actually said something: `merge_profile`
    # already carried forward whatever was set on an earlier turn, and
    # unioning old and new values here (as `merge_profile` would for a list
    # field) would leave "Veg" and "Non-Veg" both sitting in the list after a
    # single "make it non-veg" -- the newest explicit statement has to win
    # outright, not get added alongside the old one.
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

        # A field is never asked twice. If the reply couldn't be used we move
        # on rather than repeating ourselves until the budget runs out.
        asked.append(field)
        # Both sides of the exchange go into `messages`: a bare reply like
        # "2" or "Indian" is meaningless to any later reader without the
        # question that prompted it. Recorded as typed, before any option
        # resolution, so the transcript always shows what the user actually
        # sent.
        transcript.append(AIMessage(content=question, name="intent_agent"))
        transcript.append(HumanMessage(content=str(raw_answer)))

        # A digit or the option's own label/value resolves to that option's
        # value, exactly as if the user had typed it out -- this is what lets
        # them answer with one keystroke. Anything the menu didn't
        # anticipate (including no options at all) passes through unchanged,
        # so free text keeps working exactly as before options existed.
        answer = resolve_option_choice(str(raw_answer), options)
        if answer == OTHER_VALUE:
            # They picked "type your own" rather than a suggestion -- ask
            # once more with no options so the free-text reply is unambiguous.
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
