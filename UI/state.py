"""Streamlit session-state setup for the Cravr chat UI.

Keeps every `st.session_state` key the app depends on in one place, so the
other UI modules never invent a key name that drifts from what's actually
initialised here.
"""

from __future__ import annotations

import uuid

import streamlit as st


def init_state() -> None:
    """Populate any session-state keys missing on this run (idempotent)."""
    if "history" not in st.session_state:
        st.session_state.history = []  # list[dict]: role, content, source?, options?, dishes?
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    if "awaiting_interrupt" not in st.session_state:
        st.session_state.awaiting_interrupt = False
    if "started" not in st.session_state:
        st.session_state.started = False
    if "error" not in st.session_state:
        st.session_state.error = None


def reset_state() -> None:
    """Drop the current conversation so the next rerun starts a fresh thread."""
    for key in ("history", "config", "awaiting_interrupt", "started", "error"):
        st.session_state.pop(key, None)
