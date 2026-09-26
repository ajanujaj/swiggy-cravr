from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from .debug_utils import debug
from .llm import get_llm
from .schemas import SuggestedOptions

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
