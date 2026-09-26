"""The app's single notion of "now": India Standard Time (UTC+05:30).

Meal windows ("It's 10:04 PM -- ...?"), breakfast/lunch/dinner defaults and
restaurant open/closed behaviour are all tied to the user's local clock. Hosts
(Render, containers, CI) run in UTC, so a bare `datetime.now()` there is 5h30m
behind and picks the wrong meal. India has no DST, so a fixed offset is exact
and needs no tzdata package.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def now_ist() -> datetime:
    return datetime.now(IST)
