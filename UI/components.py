from __future__ import annotations

import json
import re
from typing import Any

import chainlit as cl

from cravr.options import OTHER_VALUE

import state

_HINT = "Or type your own answer below."
_NONE_ADDON_VALUE = "__no_addon__"
_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+")
_LIST_THEN_TEXT = re.compile(r"^(\s*[-*] .*)\n(?!\s*[-*] |\s*$)(.*)$", re.M)
_PHRASE_FIXES = [
    (re.compile(r"Pick one by number or name, ", re.I), "Tap one, "),
    (re.compile(r"\s*\((?:numbers, names, or comma-separated)\)", re.I), ""),
    (re.compile(r"\s*\(numbers, names, or comma-separated\), or say \"none\"\.", re.I), "."),
    (re.compile(r"select as many as you like", re.I), "pick as many as you like"),
]


def _clean_text(content: str, has_options: bool) -> str:
    text = content or ""
    if has_options:
        text = "\n".join(line for line in text.split("\n") if not _NUMBERED_LINE.match(line))
        for pattern, replacement in _PHRASE_FIXES:
            text = pattern.sub(replacement, text)
    text = _LIST_THEN_TEXT.sub(r"\1\n\n\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _chip_options(options: list[dict[str, str]] | None) -> list[dict[str, Any]]:
    return [
        {
            "value": str(o.get("value")),
            "label": str(o.get("label") or o.get("value") or ""),
            "group": o.get("group") or "",
            "required": bool(o.get("required")),
        }
        for o in options or []
        if o.get("value") not in (OTHER_VALUE, _NONE_ADDON_VALUE, None)
    ]


def _element(name: str, props: dict[str, Any]) -> cl.CustomElement:
    return cl.CustomElement(name=name, props=props, display="inline")


async def _send(content: str, element: cl.CustomElement | None) -> cl.Message:
    message = cl.Message(content=content, elements=[element] if element else [], author="Cravr")
    await message.send()
    return message


async def send_option_message(question: str, options: list[dict[str, str]] | None) -> None:
    chips = _chip_options(options)
    element = _element("OptionChips", {"options": chips, "hint": _HINT}) if chips else None
    await _send(_clean_text(question, bool(chips)), element)
    state.set_live_elements([element] if element else [])


async def send_assistant_message(content: str, options: list[dict[str, str]] | None) -> None:
    await send_option_message(content, options)


def _dish_props(dish: dict[str, Any], option: dict[str, Any] | None) -> dict[str, Any]:
    name = str(dish.get("item_name") or "Dish")
    diet = str(dish.get("dietary_type") or "").lower()
    tags = [t.strip() for t in re.split(r"[;,]", str(dish.get("tags") or "")) if t.strip()]
    if option:
        option = {**option, "cta": "Pick this" if option["label"] == name else option["label"]}
    return {
        "name": name,
        "restaurant": dish.get("restaurant_name") or "",
        "cuisine": dish.get("cuisine") or "",
        "price": dish.get("price_inr") or None,
        "rating": dish.get("rating"),
        "tags": [t for t in tags if t.lower() not in ("vegetarian", "veg", "non-veg")],
        "veg": True if diet == "veg" else False if diet in ("non-veg", "nonveg") else None,
        "option": option,
    }


def _intro_text(content: str, dishes: list[dict[str, Any]]) -> str:
    names = {str(d.get("item_name") or "").strip().lower() for d in dishes} - {""}
    kept = [
        line for line in (content or "").split("\n")
        if not (_NUMBERED_LINE.match(line) and any(n in line.lower() for n in names))
    ]
    text = _clean_text("\n".join(kept), True)
    if len(dishes) == 1:
        text = re.sub(r"^I’d suggest .*? — that's the only match for what you asked\.?",
                      "Only one dish matches what you asked for:", text)
    return text


async def send_dish_cards(
    content: str,
    dishes: list[dict[str, Any]],
    pick_options: list[dict[str, str]] | None,
) -> None:
    chips = _chip_options(pick_options)
    cards = [_dish_props(d, chips[i] if i < len(chips) else None) for i, d in enumerate(dishes)]
    element = _element("DishCards", {"dishes": cards, "extras": chips[len(dishes):], "hint": _HINT})
    await _send(_intro_text(content, dishes), element)
    state.set_live_elements([element] if chips else [])


def _multiselect_question(question: str) -> str:
    return _clean_text(question.split("\n", 1)[0], True)


async def send_multiselect_message(question: str, options: list[dict[str, str]] | None) -> None:
    chips = _chip_options(options)
    element = _element("OptionChips", {"options": chips, "multi": True, "hint": _HINT})
    await _send(_multiselect_question(question), element)
    state.set_multiselect({"element": element, "options": chips})
    state.set_live_elements([element])


def multiselect_answer(values: list[str]) -> tuple[str, str]:
    current = state.get_multiselect() or {}
    state.set_multiselect(None)
    if not values:
        return _NONE_ADDON_VALUE, "None of these"
    labels = {o["value"]: o["label"] for o in current.get("options") or []}
    return ",".join(values), ", ".join(labels.get(v, v) for v in values)


async def lock_element(element: cl.CustomElement, selected: Any = None) -> None:
    element.props = {**element.props, "locked": True, "selected": selected}
    element.content = json.dumps(element.props)
    await element.update()
