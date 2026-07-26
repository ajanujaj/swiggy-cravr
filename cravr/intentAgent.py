"""Intent Discovery agent: turns a food-ordering request into structured context."""

from __future__ import annotations

import re

from langchain_core.messages import SystemMessage

from .debug_utils import debug
from .llm import get_llm
from .profile_utils import latest_user_text, merge_profile
from .schemas import IntentExtraction
from .state import CravrState


INTENT_DISCOVERY_PROMPT = """
You are Cravr's Intent Discovery agent. Analyze the user's food-ordering message(s).
You are not a conversational assistant: do not recommend food and do not ask a
question. Extract only facts explicitly stated or safely implied by the message.
Use null/empty values for anything not mentioned. Never fabricate values.

For taste_preference, map the user's wording onto one or more of exactly these
tags: spicy, mild, sweet, savory, tangy. Users describe taste in many
different ways -- match the intent, not just the literal word. Examples:
- spicy: "spicy", "hot", "extra hot", "fiery", "with chilli/chili", "make it hot"
- mild: "mild", "not spicy", "less spicy", "no chilli", "easy on the spice",
  "bland is fine", "nothing too hot"
- sweet: "sweet", "sweet tooth", "something sugary", "dessert-like", "sweet treat"
- savory: "savory", "salty", "umami", "hearty", "something filling and savory"
- tangy: "tangy", "tart", "sour", "citrusy", "zesty"
A message can carry more than one (e.g. "sweet and tangy" -> ["sweet", "tangy"]).
If taste isn't mentioned or implied at all, leave taste_preference empty --
never guess a taste preference the user didn't express.
""".strip()


def _important_missing_fields(profile: dict) -> list[str]:
    """Decide what's still missing from the MERGED profile, not from the
    model's own (unreliable, per-turn) self-report. A small local model isn't
    trustworthy about remembering the whole conversation, so this checks the
    accumulated fields directly instead.

    Only genuine blockers belong here -- every entry costs the user a question
    before they see any food. Diet blocks, because suggesting chicken to a
    vegetarian is actively wrong. Budget does not: it's an optional refinement,
    used when volunteered, and demanding it up front is friction that rarely
    changes the answer (spicy mains span ₹180-520).
    """
    missing: list[str] = []
    if not _has_flavour_signal(profile):
        missing.append("craving_or_cuisine")
    if not profile.get("dietary_preferences"):
        missing.append("dietary_preferences")
    return missing


def _has_flavour_signal(profile: dict) -> bool:
    """True when the user has told us *anything* we can steer a menu with.

    A taste counts. "I want something spicy" IS a craving -- it narrows the
    menu to 18 mains across 6 cuisines, which is plenty to recommend from.
    Checking only `craving`/`cuisine` here is what made Cravr answer "I want
    something spicy" with "do you have a specific craving in mind?".
    """
    return bool(
        profile.get("craving") or profile.get("cuisine") or profile.get("taste_preference")
    )


# Cuisine words the MCP menu can actually filter on. The tool matches with
# `str.contains`, so "indian" correctly hits both "North Indian" and
# "South Indian" -- no normalization to an exact label is needed.
KNOWN_CUISINE_WORDS = (
    "north indian", "south indian", "indian", "chinese", "italian", "mexican",
    "japanese", "thai", "continental", "punjabi", "mughlai", "asian",
)

# Answers that mean "I don't care" -- treated as *resolved*, not as a value.
# Without this, "anything" would be stored as a literal cuisine and every
# downstream filter would return nothing.
NO_PREFERENCE_ANSWERS = (
    "anything", "any", "no preference", "nothing", "none", "doesn't matter",
    "does not matter", "dont matter", "don't matter", "you decide", "you choose",
    "surprise me", "whatever", "up to you", "no idea", "not sure", "idk",
)

# The five taste tags menu.csv actually carries, and the words users reach for.
# Answering "something spicy" must land in `taste_preference` (a real,
# filterable tag) rather than `craving` -- a free-text craving is dropped when
# building menu filters, so the user's whole stated preference evaporates.
TASTE_CUES = {
    "spicy": ("spicy", "hot", "fiery", "chilli", "chili", "masala"),
    "sweet": ("sweet", "sugary", "sweet tooth"),
    "savory": ("savory", "savoury", "salty", "umami", "hearty"),
    "tangy": ("tangy", "tart", "sour", "citrusy", "zesty"),
}
# Matched first, and stripped out before the table above is scanned, so that
# "nothing too spicy" reads as mild instead of matching "spicy". A pattern
# rather than a phrase list, because the ways to negate heat are endless
# ("not spicy", "nothing too spicy", "less hot", "no chilli please").
MILD_PATTERN = re.compile(
    r"\b(?:not|no|nothing|less|avoid|without)\s+(?:too\s+|very\s+)?(?:spicy|hot|chill?i)\b"
)
MILD_WORDS = ("mild", "bland")

VEG_ANSWERS = ("vegetarian", "veg", "pure veg", "no meat", "veggie")
NON_VEG_ANSWERS = ("non-veg", "non veg", "nonveg", "non-vegetarian", "non vegetarian",
                   "meat", "chicken", "mutton", "fish", "egg", "eggetarian")


def _is_no_preference(answer: str) -> bool:
    normalized = answer.strip().lower().rstrip(".!")
    return normalized in NO_PREFERENCE_ANSWERS


def extract_tastes(text: str) -> list[str]:
    """Map free wording onto the menu's five taste tags."""
    remaining = f" {text} "
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


def _apply_clarification_answer(profile: dict, field: str, answer: str) -> bool:
    """Write the user's reply straight into the field it was asked about.

    `decision_agent` knows exactly which gap it just asked the user to fill,
    so that answer should never have to survive a round-trip through the
    extraction model to land in the profile. It previously did, and a bare
    reply like "Indian" / "vegetarian" / "1000" was frequently dropped by
    `llama3.2:1b`, which is what caused the same question to be asked over
    and over. The LLM extraction still runs first and wins when it succeeds;
    this only fills in what it missed.

    Returns True when the profile was actually updated.
    """
    answer = (answer or "").strip()
    if not answer or _is_no_preference(answer):
        return False
    normalized = answer.lower()

    if field == "craving_or_cuisine":
        applied = False

        # A taste is the most filterable thing in the answer, so pull it out
        # first: "something spicy" is worth far more as taste_preference=
        # ["spicy"] than as craving="something spicy", which gets discarded
        # when the menu filters are built.
        tastes = extract_tastes(normalized)
        if tastes:
            existing = list(profile.get("taste_preference") or [])
            for taste in tastes:
                if taste not in existing:
                    existing.append(taste)
                    applied = True
            profile["taste_preference"] = existing

        if not profile.get("cuisine"):
            for word in KNOWN_CUISINE_WORDS:
                if word in normalized:
                    profile["cuisine"] = answer if len(answer) <= 30 else word.title()
                    applied = True
                    break

        # Nothing structured in there -- keep the raw wording as a craving.
        if not applied and not profile.get("craving") and not profile.get("cuisine"):
            profile["craving"] = answer
            applied = True

        return applied

    if field == "dietary_preferences":
        if profile.get("dietary_preferences"):
            return False
        if any(token in normalized for token in NON_VEG_ANSWERS):
            profile["dietary_preferences"] = ["non-vegetarian"]
            return True
        if any(token in normalized for token in VEG_ANSWERS):
            profile["dietary_preferences"] = ["vegetarian"]
            return True
        return False

    if field == "budget_inr":
        if profile.get("budget_inr") is not None:
            return False
        # "around 1,000 rupees" / "under Rs. 500" / "1000"
        match = re.search(r"\d[\d,]*", normalized)
        if match:
            try:
                profile["budget_inr"] = int(match.group().replace(",", ""))
                return True
            except ValueError:
                return False
        return False

    return False


def _known_phrase(profile: dict) -> str:
    """Describe what we already understood, e.g. "spicy Indian"."""
    bits: list[str] = []
    for taste in profile.get("taste_preference") or []:
        if taste and str(taste).lower() not in bits:
            bits.append(str(taste).lower())

    headline = profile.get("cuisine") or profile.get("craving")
    if headline and str(headline).lower() not in bits:
        bits.append(str(headline))

    return " ".join(bits)


def _question_for(item: str, profile: dict | None = None) -> str:
    """Phrase the follow-up so it's obvious we heard the user.

    A flat "should the meal be vegetarian?" straight after "I want something
    spicy" reads like the request was ignored, so lead with what we have.
    """
    profile = profile or {}

    if item == "craving_or_cuisine":
        return "What are you in the mood for — a particular dish, cuisine, or flavour?"

    if item == "dietary_preferences":
        known = _known_phrase(profile)
        if known:
            return f"{known[0].upper()}{known[1:]} it is — veg or non-veg?"
        return "Veg or non-veg?"

    if item == "budget_inr":
        return "What's your budget for this order in INR?"

    return "Can you clarify this: %s" % item


def intent_discovery_node(state: CravrState) -> dict:
    """Call the local model with a schema-enforced structured-output call,
    merge what it found onto the running intent profile (so earlier answers
    survive even if this turn's extraction doesn't restate them), and
    recompute what's still missing from the merged facts.

    If `decision_agent` just asked a targeted question (`pending_clarification_
    field` is set), the user's reply is *also* applied directly to that field
    as a fallback -- the extraction model is too small to reliably tell that
    "Indian" was an answer to "which cuisine?" rather than noise.
    """
    pending_field = state.get("pending_clarification_field")
    is_clarification_reply = bool(pending_field)

    try:
        structured_llm = get_llm().with_structured_output(IntentExtraction)
        extraction = structured_llm.invoke([SystemMessage(content=INTENT_DISCOVERY_PROMPT), *state["messages"]])
        extracted = extraction.model_dump()
        debug("intent_agent", f"structured extraction={extracted}")
    except Exception as exc:
        debug("intent_agent", f"structured output call failed ({exc}) — treating as no new information")
        extracted = {}

    profile = merge_profile(state.get("intent_profile"), extracted)

    if is_clarification_reply:
        answer = latest_user_text(state.get("messages") or [])
        applied = _apply_clarification_answer(profile, str(pending_field), answer)
        debug(
            "intent_agent",
            f"clarification reply for {pending_field!r}: answer={answer!r} applied_directly={applied}",
        )

    important_missing = _important_missing_fields(profile)
    profile["important_missing_information"] = important_missing
    profile["clarification_questions"] = [_question_for(item, profile) for item in important_missing]

    debug("intent_agent", f"merged profile={profile}")

    updates: dict = {"intent_profile": profile, "pending_clarification_field": None}
    if not is_clarification_reply:
        # A brand-new user turn (not a clarify-and-retry loop) starts a fresh
        # clarification streak. Without this reset the counter only ever went
        # back to 0 on give-up, so an unanswerable question got re-asked in
        # streak after streak instead of being dropped.
        updates["clarification_rounds"] = 0
        updates["asked_clarification_fields"] = []

    return updates
