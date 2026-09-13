"""Generic LLM-backed multiple-choice suggestions for clarification questions.
`suggest_options()` turns a free-text clarification question into a short,
tappable menu; `resolve_option_choice()` maps the reply (a digit, an
option's own text, or unanticipated free text) back into a plain value.
Any `interrupt({"question": q, "options": []})` call can switch to
`suggest_options(q, ...)` to get the same treatment.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from .debug_utils import debug
from .llm import get_llm
from .schemas import SuggestedOptions

# Sentinel for the "type your own" option. Never a real field value -- callers must catch it
# and ask a follow-up free-text question instead.
OTHER_VALUE = "__other__"
OTHER_LABEL = "Something else (type your own)"

_OPTIONS_PROMPT = """
You are Cravr's option-suggestion helper. Cravr is a food-ordering assistant
for someone who isn't sure what to eat, so it should ask for as little
typing as possible. Given a question Cravr is about to ask and whatever is
already known about the request, propose up to 5 short (1-4 word), concrete,
mutually exclusive answer choices a busy person could tap instead of typing
a sentence -- aim for 5 whenever the context genuinely supports that many
distinct, plausible answers; offer fewer only when it doesn't (never pad
with a weak or redundant choice just to hit 5). Tailor them tightly to the
context given -- meal time, meal already chosen, hunger level,
cuisine/craving hints -- so a breakfast question and a late-night question
about the same field get visibly different, time-appropriate choices rather
than the same generic list. Never include a catch-all choice like "other" or
"something else" -- that is added separately. Never invent facts about the
user; base choices only on the question and the given context.
""".strip()


def suggest_options(
    question: str,
    context: dict | None = None,
    fallback: list[str] | None = None,
    include_other: bool = True,
    max_options: int = 5,
) -> list[dict[str, str]]:
    """LLM-suggested tappable answers for `question`, with a static fallback.
    Returns interrupt-payload option dicts (`{"value", "label"}`), ending with
    an "other" escape hatch by default. Never raises -- any model failure falls
    back to `fallback` so a clarification question is never blocked on the LLM.
    """
    labels: list[str] = []
    try:
        structured_llm = get_llm().with_structured_output(SuggestedOptions)
        payload = {"question": question, "known_so_far": context or {}}
        result = structured_llm.invoke([
            SystemMessage(content=_OPTIONS_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        labels = [
            label.strip()
            for label in (getattr(result, "options", None) or [])
            if isinstance(label, str) and label.strip()
        ]
        debug("options", f"LLM suggested options for {question!r}: {labels}")
    except Exception as exc:
        debug("options", f"option suggestion failed ({exc}) -- using fallback")

    if not labels:
        labels = list(fallback or [])

    # De-dupe, preserving order -- the model occasionally repeats itself.
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        key = label.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(label)

    options = [{"value": label, "label": label} for label in deduped[:max_options]]
    if include_other:
        options.append({"value": OTHER_VALUE, "label": OTHER_LABEL})
    return options


def resolve_option_choice(answer: str, options: list[dict]) -> str:
    """Map a numbered or typed-out reply onto the option it picked. Falls through to the
    raw answer unchanged when no options were offered or none match -- free text the menu
    didn't anticipate must keep working as before options existed.
    """
    if not options:
        return answer
    normalized = (answer or "").strip().lower().rstrip(".!")
    if not normalized:
        return answer

    if normalized.isdigit():
        position = int(normalized)
        if 1 <= position <= len(options):
            return options[position - 1].get("value", answer)
        return answer

    for option in options:
        value = str(option.get("value", "")).strip().lower()
        label = str(option.get("label", "")).strip().lower()
        if normalized in {value, label}:
            return option.get("value", answer)

    return answer
