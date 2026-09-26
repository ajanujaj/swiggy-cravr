from __future__ import annotations

import re
from typing import Any

import chainlit as cl

from cravr.options import OTHER_VALUE

import state

_TAP_YOUR_OWN_CAPTION = "_Or type your own answer below._"


def _build_actions(options: list[dict[str, str]]) -> list[cl.Action]:
    return [
        cl.Action(
            name="pick",
            payload={"value": o.get("value"), "label": o.get("label") or o.get("value")},
            label=o.get("label") or o.get("value") or "",
        )
        for o in options
        if o.get("value") != OTHER_VALUE
    ]


async def send_option_message(question: str, options: list[dict[str, str]] | None) -> None:
    actions = _build_actions(options or [])
    content = f"{question}\n\n{_TAP_YOUR_OWN_CAPTION}" if actions else question
    message = cl.Message(content=content, actions=actions, author="Cravr")
    await message.send()
    state.set_live_option_messages([message] if actions else [])


def _describe(dish: dict[str, Any]) -> str:
    tags = str(dish.get("tags") or "").replace(";", " · ")
    price = dish.get("price_inr")
    rating = dish.get("rating")
    meta_bits = [
        b
        for b in (
            dish.get("cuisine"),
            f"₹{price}" if price is not None else None,
            f"⭐ {rating}" if rating is not None else None,
        )
        if b
    ]
    lines = [f"**{dish.get('item_name', 'Dish')}**"]
    if dish.get("restaurant_name"):
        lines.append(str(dish["restaurant_name"]))
    if meta_bits:
        lines.append(" · ".join(meta_bits))
    if tags:
        lines.append(f"_{tags}_")
    return "\n".join(lines)


def _strip_dish_lines(content: str, dishes: list[dict[str, Any]]) -> str:
    names = {str(d.get("item_name") or "").strip().lower() for d in dishes if d.get("item_name")}
    names.discard("")
    if not names:
        return content

    kept = []
    for line in content.split("\n"):
        stripped = line.strip()
        is_numbered = bool(re.match(r"^\d+\.\s", stripped))
        mentions_dish = any(name in stripped.lower() for name in names)
        if is_numbered and mentions_dish:
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_CARD_VIEW_MAX_DISHES = 3


async def send_dish_list(content: str, pick_options: list[dict[str, str]] | None) -> None:
    actions = _build_actions(pick_options or [])
    message = cl.Message(content=content, actions=actions, author="Cravr")
    await message.send()
    state.set_live_option_messages([message] if actions else [])


async def send_dish_cards(
    content: str,
    dishes: list[dict[str, Any]],
    pick_options: list[dict[str, str]] | None,
) -> None:
    if len(dishes) > _CARD_VIEW_MAX_DISHES:
        await send_dish_list(content, pick_options)
        return

    intro = _strip_dish_lines(content, dishes)
    if intro:
        await cl.Message(content=intro, author="Cravr").send()

    pick_options = pick_options or []
    live: list[cl.Message] = []

    for position, dish in enumerate(dishes, start=1):
        option = pick_options[position - 1] if position <= len(pick_options) else None
        actions = _build_actions([option]) if option else []
        message = cl.Message(content=_describe(dish), actions=actions, author="Cravr")
        await message.send()
        if actions:
            live.append(message)

    leftover = pick_options[len(dishes):]
    if leftover:
        actions = _build_actions(leftover)
        message = cl.Message(content=_TAP_YOUR_OWN_CAPTION, actions=actions, author="Cravr")
        await message.send()
        live.append(message)
    elif live:
        await cl.Message(content=_TAP_YOUR_OWN_CAPTION, author="Cravr").send()

    state.set_live_option_messages(live)


async def send_assistant_message(content: str, options: list[dict[str, str]] | None) -> None:
    await send_option_message(content, options)


_DONE_VALUE = "__multiselect_done__"


def _render_multiselect(question: str, options: list[dict[str, str]], selected: list[str]) -> tuple[str, list[cl.Action]]:
    chosen = set(selected)
    lines = [question.split("\n", 1)[0], ""]
    actions: list[cl.Action] = []
    current_group: str | None = None
    for opt in options:
        value = opt.get("value")
        if value in (OTHER_VALUE, "__no_addon__"):
            continue
        group = opt.get("group")
        if group and group != current_group:
            current_group = group
            if len(lines) > 2:
                lines.append("")
            required = " — *required*" if opt.get("required") else ""
            lines += [f"**{group}**{required}", ""]
        mark = "☑" if value in chosen else "☐"
        lines.append(f"- {mark} {opt.get('label') or value}")
        actions.append(cl.Action(
            name="multiselect_toggle",
            payload={"value": value},
            label=f"{'✓ ' if value in chosen else ''}{opt.get('label') or value}",
        ))
    count = len(chosen)
    lines += ["", f"_{count} selected -- tap items to toggle, then press Done. You can also type numbers or names._"]
    actions.append(cl.Action(
        name="multiselect_done", payload={"value": _DONE_VALUE},
        label=f"Done ({count})" if count else "Skip -- none of these",
    ))
    return "\n".join(lines), actions


async def send_multiselect_message(question: str, options: list[dict[str, str]] | None) -> None:
    options = options or []
    content, actions = _render_multiselect(question, options, [])
    message = cl.Message(content=content, actions=actions, author="Cravr")
    await message.send()
    state.set_multiselect({"message": message, "question": question, "options": options, "selected": []})
    state.set_live_option_messages([])


async def toggle_multiselect(value: str) -> None:
    current = state.get_multiselect()
    if not current:
        return
    selected: list[str] = current["selected"]
    if value in selected:
        selected.remove(value)
    else:
        selected.append(value)
    message: cl.Message = current["message"]
    await message.remove_actions()
    message.content, message.actions = _render_multiselect(current["question"], current["options"], selected)
    await message.update()


async def finish_multiselect() -> tuple[str, str] | None:
    current = state.get_multiselect()
    if not current:
        return None
    message: cl.Message = current["message"]
    await message.remove_actions()
    state.set_multiselect(None)
    selected: list[str] = current["selected"]
    if not selected:
        return "__no_addon__", "None of these"
    labels = {o.get("value"): o.get("label") or o.get("value") for o in current["options"]}
    return ",".join(selected), ", ".join(str(labels.get(v, v)) for v in selected)
