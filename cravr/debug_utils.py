"""Lightweight, opt-in tracing for the agent pipeline. Set `CRAVR_DEBUG=1`
to print what each node saw and decided -- raw LLM output, merged profiles,
routing choices -- to stderr. Fastest way to tell "the model returned
garbage" apart from "the code picked the wrong branch"."""

from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.getenv("CRAVR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug(source: str, message: str) -> None:
    if debug_enabled():
        print(f"[debug:{source}] {message}", file=sys.stderr)
