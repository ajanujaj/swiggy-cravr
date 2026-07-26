"""Context Discovery agent: turns a food-ordering request into structured context."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from .debug_utils import debug
from .llm import get_llm
from .profile_utils import merge_profile
from .schemas import ContextExtraction
from .state import CravrState


CONTEXT_DISCOVERY_PROMPT = """
You are Cravr's Context discovery agent. Analyze the user's food-ordering message(s).
You are not a conversational assistant: do not recommend food and do not ask a
question. Extract only facts explicitly stated or safely implied by the message.
Use null values for anything not mentioned. Never fabricate values.
""".strip()


def _important_missing_fields(profile: dict) -> list[str]:
    """Decide what's still missing from the MERGED profile, not from the
    model's own (unreliable, per-turn) self-report."""
    missing: list[str] = []
    if not profile.get("location"):
        missing.append("location")
    if not profile.get("weather"):
        missing.append("weather")
    return missing


def _question_for(item: str) -> str:
    if item == "location":
        return "What's the delivery location (city/area)?"
    if item == "weather":
        return "Is the weather (e.g., raining) affecting delivery where you are?"
    return "Can you clarify this: %s" % item


def context_discovery_node(state: CravrState) -> dict:
    """Call the local model with a schema-enforced structured-output call,
    merge what it found onto the running context profile (so an earlier
    answer survives even if this turn's extraction doesn't restate it), and
    recompute what's still missing from the merged facts.
    """
    try:
        structured_llm = get_llm().with_structured_output(ContextExtraction)
        extraction = structured_llm.invoke([SystemMessage(content=CONTEXT_DISCOVERY_PROMPT), *state["messages"]])
        extracted = extraction.model_dump()
        debug("context_agent", f"structured extraction={extracted}")
    except Exception as exc:
        debug("context_agent", f"structured output call failed ({exc}) — treating as no new information")
        extracted = {}

    profile = merge_profile(state.get("context_profile"), extracted)

    important_missing = _important_missing_fields(profile)
    profile["important_missing_information"] = important_missing
    profile["clarification_questions"] = [_question_for(item) for item in important_missing]

    debug("context_agent", f"merged profile={profile}")

    return {"context_profile": profile}
