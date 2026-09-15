"""Chat message builders for the Cravr chat UI: assistant replies with
tappable options, and dish cards. Each function sends its message(s)
directly; anything with a live action button is registered via
`state.set_live_option_messages` so the next turn can clear it.
"""

from __future__ import annotations

import re
from typing import Any

import chainlit as cl

from cravr.options import OTHER_VALUE

import state

_TAP_YOUR_OWN_CAPTION = "_Or type your own answer below._"


def _build_actions(options: list[dict[str, str]]) -> list[cl.Action]:
    """One `cl.Action` per option, skipping the "type your own" sentinel.
    All options share the `"pick"` action name; the value/label lives in
    the payload."""
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
    """Drop numbered dish-description lines from the intro text since the
    dishes render as their own cards right below. Ported from the
    Streamlit version: only lines naming a dish in `dishes` are removed."""
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


# Above this many dishes, one-bubble-per-dish stops being readable -- N
# near-identical full-width cards flood the main chat and its one pick
# button per card is easy to lose track of scrolling past them (reported
# live: "lost the button to select the options" on a 10-item list). Past
# this count we consolidate into a single message instead -- see
# send_dish_list.
_CARD_VIEW_MAX_DISHES = 3


async def send_dish_list(content: str, pick_options: list[dict[str, str]] | None) -> None:
    """A big batch of dishes, as ONE message: `content` already has the
    full numbered list as text (recommendation_agent's own formatting), and
    every pick button is attached to that single message rather than
    scattered one-per-card across N separate bubbles. Multiple `cl.Action`s
    on one message is the same, already-working pattern `send_option_message`
    uses for the intent-clarification chips -- just applied to a longer
    list, so the buttons stay grouped with the list they belong to instead
    of getting lost."""
    actions = _build_actions(pick_options or [])
    message = cl.Message(content=content, actions=actions, author="Cravr")
    await message.send()
    state.set_live_option_messages([message] if actions else [])


async def send_dish_cards(
    content: str,
    dishes: list[dict[str, Any]],
    pick_options: list[dict[str, str]] | None,
) -> None:
    """One message per dish, each with its own pick button -- only for a
    small batch (see `_CARD_VIEW_MAX_DISHES`); a longer list goes through
    `send_dish_list` instead. `pick_options` lines up with `dishes` by
    position; any extra option (e.g. a single-match "widen the search"
    choice) renders as its own message."""
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
