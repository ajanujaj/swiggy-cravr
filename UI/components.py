"""Chat message builders for the Cravr chat UI: assistant replies with
tappable options, and dish cards.

Every function here SENDS its message(s) directly (`await ....send()`)
rather than building content for a separate render pass -- Chainlit is
message-stream-native, unlike Streamlit's rerun-the-whole-script model the
old `UI/components.py` was built around, so there's no "render history"
step here at all: a message is sent once, the moment it exists, and stays
in the transcript exactly as sent.

Every message that carries a live option (an action button) is handed to
`state.set_live_option_messages(...)` so the *next* turn can clear it (see
`state.clear_live_options`) -- the mechanism that keeps only the newest
turn's buttons ever clickable, the same guarantee the old Streamlit UI got
for free by only rendering the last history entry's options on each rerun.
"""

from __future__ import annotations

import re
from typing import Any

import chainlit as cl

from cravr.options import OTHER_VALUE

import state

_TAP_YOUR_OWN_CAPTION = "_Or type your own answer below._"


def _build_actions(options: list[dict[str, str]]) -> list[cl.Action]:
    """One `cl.Action` per option, skipping the "type your own" sentinel --
    that one isn't a real answer to resume the graph with, it's a cue that
    free text is welcome, which the chat input already offers every turn.

    Every option funnels through the single `"pick"` action name (see
    `chainlit_app.on_pick`) -- the value/label actually being picked lives
    in the payload, not the action name, so a brand-new kind of question
    never needs a new `@cl.action_callback` wired up for it.
    """
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
    """A plain clarification question, with tappable answers attached
    directly to the same message -- Chainlit natively supports several
    actions on one message, so (unlike the dish-card case) there's no
    "buttons vs. content" duplication to merge here."""
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
    """Drop the numbered dish-description lines from the intro text -- the
    dishes are about to render as their own cards right below, so leaving
    these in would show the same three dishes twice. Ported unchanged from
    the Streamlit version: only lines that actually name one of `dishes`
    are removed, so unrelated numbered lines (e.g. a single-match
    confirmation's "1. Yes, I'll take it / 2. Widen the search") are left
    untouched -- those aren't dish descriptions and have no card to
    duplicate.
    """
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
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse gaps the removal left behind
    return text.strip()


async def send_dish_cards(
    content: str,
    dishes: list[dict[str, Any]],
    pick_options: list[dict[str, str]] | None,
) -> None:
    """One message per dish -- each message *is* the card, with its own
    pick button directly beneath it. Chainlit attaches actions to the
    specific message they're sent on, so this needs none of the
    CSS-overlay trick the Streamlit version needed to make a whole card
    clickable -- native behaviour here, not a workaround.

    `pick_options` lines up with `dishes` by position, exactly like
    `cravr.recommend.RecommendAgent._dish_pick_options` built it; any
    option beyond `len(dishes)` (the single-match "widen the search"
    choice, which isn't a dish at all) renders as its own small message
    underneath instead of a re-numbered dish card.
    """
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
        # No leftover message to carry the caption -- add it as its own
        # small closing note rather than silently dropping it when every
        # option offered is a dish card.
        await cl.Message(content=_TAP_YOUR_OWN_CAPTION, author="Cravr").send()

    state.set_live_option_messages(live)


async def send_assistant_message(content: str, options: list[dict[str, str]] | None) -> None:
    """A plain assistant reply -- no dishes, optionally with its own
    tappable options. Kept general to match the old Streamlit UI's
    handling, even though in practice every `options` payload today comes
    bundled with dishes (see `graph_bridge.process_result`); a future
    recommendation-agent reply that sets options without dishes is still
    handled correctly rather than silently dropping its buttons.
    """
    await send_option_message(content, options)
