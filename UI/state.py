from __future__ import annotations

import uuid
from typing import Any

import chainlit as cl


_THREAD_CONFIG_KEY = "cravr_config"
_AWAITING_INTERRUPT_KEY = "cravr_awaiting_interrupt"
_LIVE_OPTION_MESSAGES_KEY = "cravr_live_option_messages"
_MULTISELECT_KEY = "cravr_multiselect"


def init_state() -> None:
    cl.user_session.set(_THREAD_CONFIG_KEY, {"configurable": {"thread_id": str(uuid.uuid4())}})
    cl.user_session.set(_AWAITING_INTERRUPT_KEY, False)
    cl.user_session.set(_LIVE_OPTION_MESSAGES_KEY, [])
    cl.user_session.set(_MULTISELECT_KEY, None)


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
    for message in get_live_option_messages():
        try:
            await message.remove_actions()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
    set_live_option_messages([])


def get_multiselect() -> dict[str, Any] | None:
    return cl.user_session.get(_MULTISELECT_KEY)


def set_multiselect(value: dict[str, Any] | None) -> None:
    cl.user_session.set(_MULTISELECT_KEY, value)
