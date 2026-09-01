"""Keep the chat pinned to the newest message as the conversation grows.

Streamlit reruns the whole script on every interaction and, by default,
leaves the browser's scroll position wherever it was -- so a growing chat
just drifts off the bottom of the page instead of staying anchored near the
input box the way a normal chat app does. There's no first-class Streamlit
API for this; the standard workaround (used widely in the Streamlit
community) is to drop in a tiny invisible HTML component whose script reaches
out to the parent page and scrolls it to the bottom. `components.html` runs
in a real (same-origin) iframe, so its `<script>` actually executes --
unlike a `<script>` tag passed through `st.markdown`, which the browser
never runs.
"""

from __future__ import annotations

import streamlit.components.v1 as components

_SCROLL_JS = """
<script>
(function () {
    const doc = window.parent.document;
    const candidates = [
        doc.querySelector('[data-testid="stAppViewContainer"]'),
        doc.querySelector('section.main'),
        doc.body,
        doc.documentElement,
    ].filter(Boolean);
    for (const el of candidates) {
        el.scrollTop = el.scrollHeight;
    }
    window.parent.scrollTo(0, doc.body.scrollHeight);
})();
</script>
"""


def to_bottom() -> None:
    """Call once, after the chat history is rendered, on every rerun."""
    components.html(_SCROLL_JS, height=0)
