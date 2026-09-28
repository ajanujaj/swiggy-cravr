# `cravr/decision/` — freeze the brief

Sits between intent discovery and recommendation. It **always recommends** and never asks the user anything. The only `interrupt()` in the request path lives in `intent_discovery`.

## `decisionAgent.py`

| Function | Usage |
|---|---|
| `decision_agent_node(state)` | **Graph node `decision_agent`.** Reads the intent, history and preference profiles and calls `recommend.build_recommendation_brief(...)`. Returns `recommendation_brief`, `summary`, `ready_to_recommend=True`, `decision="recommend"` and a short internal message. |
| `_coerce_profile(profile)` | Turns a profile that may be a dict, Pydantic model or `None` into a plain dict. |
| `_extract_latest_user_input(messages)` | Latest user text (for debug logging). |

The **brief** is the frozen hand-off: filters (cuisine, tags, dietary type, budget, course, craving) plus a history summary. It is built exactly once here so the recommender never re-derives it.

## `__init__.py`
Re-exports `decision_agent_node`.
