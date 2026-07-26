"""Lightweight, opt-in tracing for the agent pipeline.

Set `CRAVR_DEBUG=1` before running (`export CRAVR_DEBUG=1` or
`CRAVR_DEBUG=1 python -m cravr.run`) to print what each node actually saw
and decided -- raw LLM output, merged profiles, and routing choices -- to
stderr. This is the fastest way to tell "the model returned garbage" apart
from "the code's logic picked the wrong branch" when a live local model
behaves unexpectedly.
"""

from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.getenv("CRAVR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug(source: str, message: str) -> None:
    if debug_enabled():
        print(f"[debug:{source}] {message}", file=sys.stderr)
