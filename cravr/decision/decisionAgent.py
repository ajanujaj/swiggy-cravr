"""Decision agent: uses the LLM to decide whether the request is ready for recommendations.

This node never calls `interrupt()` itself (see `DecisionAgentNeverAsksTests`
in tests/test_decision_agent.py) -- only intent_discovery_node does, so the
whole graph has exactly one interrupt site and one clarification round
budget, never two competing ones. When decision_agent_node wants to ask
something intent didn't already cover (e.g. a saved-preference conflict like
"vegetarian or non-vegetarian?"), it hands the question off through
`pending_decision_question` and routes back to intent_discovery_node via
`CLARIFY_AND_RETRY`; intent_discovery_node asks it with tappable options the
same way it asks its own fields, then folds the answer back in through the
normal extraction pipeline before decision_agent_node is asked again.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..debug_utils import debug
from ..llm import get_llm
from ..profile_utils import latest_user_text
from ..recommend import build_recommendation_brief
from ..schemas import DecisionOutput
from ..state import CravrState
from .prompts import DECISION_PROMPT, GIVE_UP_MESSAGE, GENERIC_CLARIFICATION_FALLBACK
from .utils import  CLARIFY_AND_RETRY, MAX_CLARIFICATION_ROUNDS, GENERIC_CLARIFICATION_FIELD


def _coerce_profile(profile: Any) -> dict:
    if isinstance(profile, dict):
        return profile
    return {}


def _extract_latest_user_input(messages: Any) -> str:
    return latest_user_text(messages)


def _is_vague_request(user_input: str, intent_profile: dict, context_profile: dict) -> bool:
    """Return True when the request is too general to recommend confidently."""
    normalized = (user_input or "").strip().lower()
    if not normalized:
        return True

    explicit_preference_markers = [
        "spicy",
        "vegetarian",
        "vegan",
        "non-veg",
        "chinese",
        "indian",
        "italian",
        "mexican",
        "thai",
        "pizza",
        "burger",
        "biryani",
        "fried rice",
        "dessert",
        "budget",
        "under",
        "below",
        "max",
        "craving",
        "cuisine",
        "protein",
        "low carb",
        "high protein",
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "tea",
        "coffee",
        "sweet",
        "salty",
        "soup",
        "salad",
        "rice",
        "noodles",
    ]
    if any(marker in normalized for marker in explicit_preference_markers):
        return False

    generic_terms = [
        "something",
        "anything",
        "eat",
        "food",
        "meal",
        "restaurant",
        "order",
        "recommend",
        "light food",
        "healthy food",
        "quick food",
        "light",
        "healthy",
        "quick",
    ]
    if not any(term in normalized for term in generic_terms):
        return False

    missing_fields = []
    for profile in (intent_profile, context_profile):
        if not isinstance(profile, dict):
            continue
        missing_fields.extend(profile.get("missing_information") or [])
        missing_fields.extend(profile.get("important_missing_information") or [])

    missing_text = " ".join(str(item or "") for item in missing_fields).lower()
    return any(term in missing_text for term in ["craving", "cuisine", "diet", "budget", "location"])


def _build_summary(intent_profile: dict, context_profile: dict, history_profile: dict, preference_profile: dict, user_input: str) -> str:
    """Create a single concise summary that captures the user's intent, context, and order history."""
    parts: list[str] = []

    if user_input:
        parts.append(f"User asked: {user_input}")

    intent_bits = []
    if intent_profile.get("craving"):
        intent_bits.append(f"craving={intent_profile['craving']}")
    if intent_profile.get("cuisine"):
        intent_bits.append(f"cuisine={intent_profile['cuisine']}")
    dietary = intent_profile.get("dietary_preferences") or []
    if dietary:
        intent_bits.append(f"dietary_preferences={','.join(map(str, dietary))}")
    budget = intent_profile.get("budget_inr")
    if budget is not None:
        intent_bits.append(f"budget_inr={budget}")
    if intent_bits:
        parts.append("intent:" + "; ".join(intent_bits))

    context_bits = []
    if context_profile.get("location"):
        context_bits.append(f"location={context_profile['location']}")
    if context_profile.get("delivery_area"):
        context_bits.append(f"delivery_area={context_profile['delivery_area']}")
    if context_profile.get("weather"):
        context_bits.append(f"weather={context_profile['weather']}")
    if context_bits:
        parts.append("context:" + "; ".join(context_bits))

    preferences = preference_profile.get("preferences") or {}
    if isinstance(preferences, list):
        preferences = preferences[0] if preferences else {}
    preference_bits = []
    if isinstance(preferences, dict):
        if preferences.get("dietary_preference"):
            preference_bits.append(f"saved_diet={preferences['dietary_preference']}")
        if preferences.get("default_budget_inr") is not None:
            preference_bits.append(f"saved_budget_inr={preferences['default_budget_inr']}")
    if preference_bits:
        parts.append("preferences:" + "; ".join(preference_bits))

    history_bits = []
    orders = history_profile.get("orders") or []
    if orders:
        history_bits.append(f"recent_orders={len(orders)}")
        recent = orders[0]
        if recent.get("restaurant_name"):
            history_bits.append(f"last_restaurant={recent['restaurant_name']}")
        if recent.get("item_name"):
            history_bits.append(f"last_item={recent['item_name']}")
    else:
        history_bits.append("recent_orders=0")
    if history_bits:
        parts.append("history:" + "; ".join(history_bits))

    return " | ".join(parts) if parts else "No additional context available."


def _collect_clarification_targets(intent_profile: dict, context_profile: dict) -> list[tuple[str, str]]:
    """Pair each still-missing field with the follow-up question for it.

    `intentAgent`/`ContextAgent` publish `important_missing_information` (field
    names) alongside a positionally-matching `clarification_questions` list.
    Keeping them paired is what lets the node ask about a *different* field
    each round instead of pinning `questions[0]` forever -- previously, if the
    first field's answer failed to extract, the second and third questions
    were simply unreachable.
    """
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()

    for profile in (intent_profile, context_profile):
        if not isinstance(profile, dict):
            continue
        fields = [str(f) for f in (profile.get("important_missing_information") or []) if f]
        questions = [q for q in (profile.get("clarification_questions") or []) if q]

        for index, field in enumerate(fields):
            if field in seen:
                continue
            seen.add(field)
            targets.append((field, questions[index] if index < len(questions) else GENERIC_CLARIFICATION_FALLBACK))

        # Questions with no matching field name (older/partial profiles) still
        # get asked, keyed by the question text itself.
        for question in questions[len(fields):]:
            key = f"question:{question}"
            if key in seen:
                continue
            seen.add(key)
            targets.append((key, question))

    if not targets:
        # The request is vague but neither profile named a specific gap (e.g.
        # only the non-important `missing_information` was populated). Ask the
        # catch-all once rather than saying nothing.
        targets.append((GENERIC_CLARIFICATION_FIELD, GENERIC_CLARIFICATION_FALLBACK))

    return targets


def _collect_clarification_questions(intent_profile: dict, context_profile: dict) -> list[str]:
    return [question for _field, question in _collect_clarification_targets(intent_profile, context_profile)]


def _needs_clarification(user_input: str, intent_profile: dict, context_profile: dict) -> bool:
    if _is_vague_request(user_input, intent_profile, context_profile):
        return True
    return bool(
        (intent_profile.get("important_missing_information") or [])
        or (context_profile.get("important_missing_information") or [])
    )


def _has_usable_signal(intent_profile: dict) -> bool:
    """True when the profile carries at least one thing we can filter a menu on.

    Tracks `IntentExtraction`: craving/cuisine/craving_tag are the filterable
    fields. `meal_intent` alone does not count -- "dinner" narrows nothing on
    its own, since 48 of the 80 menu rows are All Day.
    """
    return bool(
        intent_profile.get("craving")
        or intent_profile.get("cuisine")
        or intent_profile.get("craving_tag")
    )


def _next_clarification(
    intent_profile: dict, context_profile: dict, already_asked: list[str]
) -> tuple[str | None, str | None]:
    """Return the first (field, question) pair not already asked this streak."""
    for field, question in _collect_clarification_targets(intent_profile, context_profile):
        if field not in already_asked:
            return field, question
    return None, None


def _next_clarification_question(intent_profile: dict, context_profile: dict) -> str:
    questions = _collect_clarification_questions(intent_profile, context_profile)
    return questions[0] if questions else GENERIC_CLARIFICATION_FALLBACK


def decision_agent_node(state: CravrState) -> dict:
    """Judge whether the bot can recommend now. This node never calls `interrupt()`.

    `intent_discovery_node` owns every actual `interrupt()` call: by the time
    control reaches here the user has already been asked about every gap
    intent could name, up to its own round budget. Calling `interrupt()` here
    too would mean two interrupt sites competing over one conversation -- the
    user answering the same gap twice, and two independent round budgets
    neither of which knows about the other.

    A gap that survives intent is therefore not automatically a reason to
    stop -- it's usually a reason to recommend loosely and let the user
    steer. The one case worth a fresh question is a *new* ambiguity intent
    had no way to name (e.g. a saved dietary preference now conflicting with
    what the user just asked for): that gets asked too, but via
    `pending_decision_question` + `CLARIFY_AND_RETRY`, so intent_discovery_node
    is still the only place `interrupt()` is ever called from, and its own
    round budget (`clarification_rounds`, capped at `MAX_CLARIFICATION_ROUNDS`)
    still applies. The only genuine dead end is a profile with nothing
    filterable in it at all once that budget is spent.
    """
    intent_profile = _coerce_profile(state.get("intent_profile"))
    context_profile = _coerce_profile(state.get("context_profile"))
    history_profile = _coerce_profile(state.get("history_profile"))
    preference_profile = _coerce_profile(state.get("preference_profile"))
    messages = state.get("messages") or []
    user_input = _extract_latest_user_input(messages)
    summary = _build_summary(
        intent_profile, context_profile, history_profile, preference_profile, user_input
    )

    debug("decision_agent", f"user_input={user_input!r}")
    debug("decision_agent", f"intent_profile={intent_profile}")
    debug("decision_agent", f"context_profile={context_profile}")
    debug("decision_agent", f"preference_profile={preference_profile}")
    # Whatever intent could not fill, it has already asked about and given up
    # on. Treat that as spent rather than as a fresh reason to interrogate.
    exhausted = bool(intent_profile.get("important_missing_information"))
    debug("decision_agent", f"gaps left by intent={intent_profile.get('important_missing_information')}")

    debug("decision_agent", f"asking the decision LLM (exhausted={exhausted})")
    payload = {
        "summary": summary,
    }
    debug("decision_agent", f"payload={payload}")
    try:
        structured_llm = get_llm().with_structured_output(DecisionOutput)
        decision_output = structured_llm.invoke([
            SystemMessage(content=DECISION_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        result = decision_output.model_dump()
        debug("decision_agent", f"structured decision={result}")
    except Exception as exc:
        debug("decision_agent", f"structured decision call failed ({exc}) — falling back to clarify")
        result = {"ready_to_recommend": False, "decision": "clarify", "clarification_message": ""}

    ready_to_recommend = bool(result.get("ready_to_recommend"))
    decision = result.get("decision") if result.get("decision") in {"recommend", "clarify"} else "clarify"

    # Decision-level round budget, separate from intent's own (tracked in
    # `intent_profile["important_missing_information"]`/`exhausted` above).
    # This is what caps *this* node's own ad hoc questions -- the ones intent
    # never asked about, like a saved-preference conflict -- at
    # MAX_CLARIFICATION_ROUNDS instead of letting the decision LLM invent a
    # new question forever.
    clarification_rounds = int(state.get("clarification_rounds") or 0)
    give_up = exhausted or clarification_rounds >= MAX_CLARIFICATION_ROUNDS

    if give_up and not ready_to_recommend and _has_usable_signal(intent_profile):
        # We've spent the clarification budget and the user did tell us
        # *something* usable. Ending the turn with another question is the
        # worst outcome; attempt a (deliberately loosely-filtered)
        # recommendation instead and let the user steer from there.
        debug("decision_agent", "clarification exhausted but profile has signal — recommending anyway")
        ready_to_recommend = True
        decision = "recommend"

    if not ready_to_recommend and not give_up:
        # Ask with tappable options instead of ending the turn on a
        # type-it-yourself question -- see the module docstring for why this
        # hands the question to intent_discovery_node rather than asking it
        # directly.
        pending_question = result.get("clarification_message") or _next_clarification_question(
            intent_profile, context_profile
        )
        debug(
            "decision_agent",
            f"handing question to intent_discovery (round {clarification_rounds + 1}/"
            f"{MAX_CLARIFICATION_ROUNDS}): {pending_question!r}",
        )
        return {
            "ready_to_recommend": False,
            "decision": CLARIFY_AND_RETRY,
            "pending_decision_question": pending_question,
            "pending_clarification_field": GENERIC_CLARIFICATION_FIELD,
            "clarification_rounds": clarification_rounds + 1,
            "summary": summary,
        }

    if give_up:
        # Out of clarification budget: never echo a question we just gave up
        # on, or the user's reply restarts the streak from scratch.
        fallback_message = GIVE_UP_MESSAGE
    else:
        fallback_message = _next_clarification_question(intent_profile, context_profile)
    clarification_message = result.get("clarification_message") or fallback_message

    updates: dict = {
        "ready_to_recommend": ready_to_recommend,
        "decision": decision,
        "messages": [
            AIMessage(
                content=clarification_message if not ready_to_recommend else "Ready to recommend.",
                name="decision_agent",
            )
        ],
        "summary": summary,
        "clarification_rounds": 0,
        "asked_clarification_fields": [],
        "pending_clarification_field": None,
        "pending_decision_question": None,
    }

    if ready_to_recommend:
        # Freeze the request exactly once, at the moment we commit to
        # recommending. Everything downstream reads this and only this.
        brief = build_recommendation_brief(
            intent_profile, history_profile, preference_profile
        )
        debug("decision_agent", f"brief filters={brief['filters']} history={brief['history']}")
        debug("decision_agent", f"brief summary={brief['summary']!r}")
        updates["recommendation_brief"] = brief
        updates["summary"] = brief["summary"]

    return updates
