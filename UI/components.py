"""Chat rendering: message bubbles, sequential option buttons, and dish cards.

Reads `st.session_state.history` (written by `graph_bridge`) and renders
it. The only place this module writes back is the option-button click
handler, which calls `graph_bridge.send` exactly the way the chat input
box does.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import streamlit as st

from graph_bridge import OTHER_VALUE, send


def _dish_card_html(position: int, dish: dict[str, Any]) -> str:
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
    return f"""
        <div class="cravr-dish-card">
            <div class="cravr-dish-name">{position}. {dish.get('item_name', 'Dish')}</div>
            <div class="cravr-dish-meta">{dish.get('restaurant_name', '')}</div>
            <div class="cravr-dish-meta">{' · '.join(meta_bits)}</div>
            <div class="cravr-dish-tags">{tags}</div>
        </div>
        """


def render_dish_cards(
    dishes: list[dict[str, Any]],
    entry_index: int | None = None,
    pick_options: list[dict[str, str]] | None = None,
) -> None:
    """Render dishes one below another, numbered to match the numbered list
    already baked into the message text (see `cravr.recommend._numbered`) --
    so "pick one by number" means the same thing whether you read the text
    or look at the cards.

    Without `pick_options` these are plain, non-interactive cards -- used
    for every dish list except the most recent one (see `render_history`).

    With `pick_options` (one `{"value", "label"}` per position, same order
    `cravr.recommend._dish_pick_options` built them in -- see
    `graph_bridge.process_result`), each card *is* the tappable answer: no
    separate "1. Matcha Pancakes" button duplicating what the card already
    shows. Streamlit can't put this card's rich HTML inside a native
    button's label, so instead a real (invisible, full-card-sized) button
    lives in the same keyed container and CSS stretches it over the card --
    see the `st-key-dish-` rules in `UI/styles.py`. If there are more
    options than dishes (e.g. the single-match "widen the search" choice,
    which isn't a dish at all), the leftover ones render as their own small
    buttons underneath, using the number already baked into their `value`
    so it still reads "2. Widen the search" and not a re-numbered "1.".
    """
    if not pick_options:
        for position, dish in enumerate(dishes, start=1):
            st.markdown(_dish_card_html(position, dish), unsafe_allow_html=True)
        return

    for position, dish in enumerate(dishes, start=1):
        option = pick_options[position - 1] if position <= len(pick_options) else None
        if option is None:
            st.markdown(_dish_card_html(position, dish), unsafe_allow_html=True)
            continue
        with st.container(key=f"dish-{entry_index}-{position}"):
            st.markdown(_dish_card_html(position, dish), unsafe_allow_html=True)
            label = dish.get("item_name") or f"Option {position}"
            if st.button(label, key=f"dish-btn-{entry_index}-{position}", use_container_width=True):
                send(option.get("value"), display_label=label)
                st.rerun()

    leftover = pick_options[len(dishes):]
    for option in leftover:
        value = option.get("value") or ""
        label = option.get("label") or value
        button_label = f"{value}. {label}" if value.isdigit() else label
        if st.button(
            button_label,
            key=f"dish-extra-{entry_index}-{value}",
            use_container_width=True,
        ):
            send(value, display_label=label)
            st.rerun()

    st.caption("Or type your own answer below.")


def _render_option_chips(entry_index: int, options: list[dict[str, str]]) -> None:
    """Tappable answer buttons for the latest clarification question.

    Stacked one below another, numbered in the same order the backend
    numbers them (see `cravr.options.suggest_options`) -- not side by side
    in columns, so the order reads top-to-bottom the same way the CLI's
    numbered list and the recommendation cards already do.

    Filters out the "type your own" sentinel (`OTHER_VALUE`) -- that one
    isn't a real answer to resume the graph with, it's a cue that free text
    is welcome, which `st.chat_input` already offers on every turn.
    """
    tappable = [o for o in options if o.get("value") != OTHER_VALUE]
    if not tappable:
        return
    for position, option in enumerate(tappable, start=1):
        label = option.get("label") or option.get("value")
        if st.button(
            f"{position}. {label}",
            key=f"opt-{entry_index}-{option.get('value')}",
            use_container_width=True,
        ):
            send(option.get("value"), display_label=label)
            st.rerun()
    st.caption("Or type your own answer below.")


def _strip_dish_lines(content: str, dishes: list[dict[str, Any]]) -> str:
    """Drop the numbered dish-description lines from the message text.

    The dishes are about to render as cards right below this text -- left
    alone, the same three dishes show up twice (once as
    "1. Chicken Panini from La Pino's Kitchen (...)" text, once as a card).
    Only lines that actually name one of `dishes` are removed (matched by
    item name), so unrelated numbered lines -- e.g. the single-match
    confirmation's "1. Yes, I'll take it / 2. Widen the search" -- are left
    untouched; those aren't dish descriptions and have no card to duplicate.
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


def render_history() -> None:
    last_index = len(st.session_state.history) - 1
    for i, entry in enumerate(st.session_state.history):
        avatar = "🧑" if entry["role"] == "user" else "🍽️"
        with st.chat_message(entry["role"], avatar=avatar):
            dishes = entry.get("dishes")
            content = _strip_dish_lines(entry["content"], dishes) if dishes else entry["content"]
            st.markdown(content)

            options = entry.get("options")
            # Only the most recent assistant turn's options are still
            # actionable -- earlier ones are just a record of what was
            # asked/offered. Not gated on `st.session_state.awaiting_interrupt`:
            # a clarification question pauses the graph on an interrupt, but
            # a dish pick (recommendation_agent's own `options`, see
            # graph_bridge.process_result) doesn't -- the graph has already
            # finished that turn, and clicking sends a normal new message.
            # `graph_bridge.send()` already branches on `awaiting_interrupt`
            # itself to resume vs. send fresh, so either kind of option
            # reaches the graph correctly regardless of this render check.
            is_actionable = bool(options) and i == last_index

            if dishes:
                render_dish_cards(dishes, entry_index=i, pick_options=options if is_actionable else None)
            elif is_actionable:
                _render_option_chips(i, options)


def render_sidebar(user_id: str, reset_state: Callable[[], None]) -> None:
    st.subheader("Session")
    st.caption(f"User: {user_id}")
    st.caption(f"Thread: {st.session_state.config['configurable']['thread_id'][:8]}")
    if st.button("Start a new conversation"):
        reset_state()
        st.rerun()
