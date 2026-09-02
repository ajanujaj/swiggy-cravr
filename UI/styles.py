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

/* Makes a whole dish card the tappable answer, instead of a plain-text
   "1. Matcha Pancakes" button duplicated underneath it. Streamlit can't put
   this card's rich HTML inside a native button's label, so
   UI/components.py's `render_dish_cards` puts the real (but invisible)
   button in the *same* keyed container as the card, and this CSS stretches
   it to cover the card exactly -- clicking anywhere on the card clicks the
   button underneath it.

   `st-key-<key>` is Streamlit's own documented, stable CSS hook for a
   `st.container(key=...)` (since Streamlit 1.32) -- not an internal
   implementation detail liable to change. Matched with a substring
   selector (`*=`) because the actual key is generated per-dish at render
   time (`dish-{entry_index}-{position}`), so this one static rule covers
   all of them. */
div[class*="st-key-dish-"] {
    position: relative;
}
div[class*="st-key-dish-"] .cravr-dish-card {
    position: relative;
    z-index: 1;
    transition: border-color 0.1s ease, background 0.1s ease;
}
/* Hover bubbles up from the invisible button (a descendant) to this
   container automatically -- plain CSS, no :has() needed -- so the card
   itself lights up to show it's clickable. */
div[class*="st-key-dish-"]:hover .cravr-dish-card {
    border-color: #ff8c42;
    background: rgba(255,140,66,0.12);
}
div[class*="st-key-dish-"] [data-testid="stButton"] {
    position: absolute;
    inset: 0;
    z-index: 2;
}
div[class*="st-key-dish-"] [data-testid="stButton"] button {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    padding: 0;
    margin: 0;
    border: none;
    opacity: 0;
    cursor: pointer;
}
</style>
"""


def inject() -> None:
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
