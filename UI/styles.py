"""CSS for the Cravr chat UI (currently just the dish cards)."""

from __future__ import annotations

import streamlit as st

_CARD_CSS = """
<style>
.cravr-dish-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    background: rgba(255,140,66,0.06);
    height: 100%;
}
.cravr-dish-name { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.15rem; }
.cravr-dish-meta { color: rgba(128,128,128,0.9); font-size: 0.85rem; }
.cravr-dish-tags { font-size: 0.78rem; color: #ff8c42; margin-top: 0.35rem; }
</style>
"""


def inject() -> None:
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
