from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.getenv("CRAVR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug(source: str, message: str) -> None:
    if debug_enabled():
        print(f"[debug:{source}] {message}", file=sys.stderr)


def test_mode_enabled() -> bool:
    return os.getenv("CRAVR_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
