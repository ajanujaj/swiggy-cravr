"""Chainlit session state for the Cravr chat UI.

Chainlit gives each connected chat session its own `cl.user_session`
key-value store -- like Streamlit's `st.session_state`, but scoped to one
websocket connection rather than reset on every script rerun. This module
is the one place that owns the keys the rest of the UI depends on, so no
other module invents a key name that drifts from what's initialised here.
"""

from __future__ import annotations

import uuid
from typing import Any

import chainlit as cl

DEFAULT_USER_ID = "USR-002"

_THREAD_CONFIG_KEY = "cravr_config"
_AWAITING_INTERRUPT_KEY = "cravr_awaiting_interrupt"
_LIVE_OPTION_MESSAGES_KEY = "cravr_live_option_messages"


def init_state() -> None:
    """Populate this session's keys. Call once, from `@cl.on_chat_start` --
    Chainlit calls that again for every new chat a person starts from its
    own UI, so a fresh LangGraph thread_id (and therefore a clean
    MemorySaver-backed conversation) falls out of that for free, the same
    way the old Streamlit sidebar's "Start a new conversation" button used
    to reset `st.session_state` by hand.
    """
    cl.user_session.set(_THREAD_CONFIG_KEY, {"configurable": {"thread_id": str(uuid.uuid4())}})
    cl.user_session.set(_AWAITING_INTERRUPT_KEY, False)
    cl.user_session.set(_LIVE_OPTION_MESSAGES_KEY, [])


def get_config() -> dict[str, Any]:
    return cl.user_session.get(_THREAD_CONFIG_KEY)


def is_awaiting_interrupt() -> bool:
    return bool(cl.user_session.get(_AWAITING_INTERRUPT_KEY))


def set_awaiting_interrupt(value: bool) -> None:
    cl.user_session.set(_AWAITING_INTERRUPT_KEY, value)


def get_live_option_messages() -> list[cl.Message]:
    return cl.user_session.get(_LIVE_OPTION_MESSAGES_KEY) or []


def set_live_option_messages(messages: list[cl.Message]) -> None:
    cl.user_session.set(_LIVE_OPTION_MESSAGES_KEY, messages)


async def clear_live_options() -> None:
    """Remove every button from the most recent turn's option message(s),
    so a superseded or already-answered question can never be tapped
    twice. Call this before handling *any* new user action (a click or a
    typed message) -- safe to call even when nothing is pending.

    Chainlit has no "only render the last history entry's buttons" concept
    the way re-rendering the whole page on every Streamlit rerun gave us
    for free -- once a message is sent with actions attached, they stay
    clickable until something explicitly removes them. This is that
    something.
    """
    for message in get_live_option_messages():
        try:
            await message.remove()
        except Exception:  # noqa: BLE001 -- best-effort cleanup; a message the client already dropped shouldn't block the new turn
            pass
    set_live_option_messages([])
