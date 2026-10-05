# `cravr/context/` — delivery context

Runs after `user_preference`, before `intent_discovery`. Adds "where are we delivering" facts to state. It is **ambient**: no LLM call, no questions to the user, and it never invents data when a lookup fails.

## `ContextAgent.py`

| Function | Usage |
|---|---|
| `context_discovery_node(state)` | **Graph node `context_discovery`.** If `context_profile` already has an `address_id` (e.g. set by `user_preference`), reuses it; otherwise fetches it with `fetch_user_location`. Merges `location`, `delivery_area` and `address_id` into the profile (via `profile_utils.merge_profile`). Returns `{"context_profile": ...}`. |
| `fetch_user_location(user_id)` | Real Swiggy: `food_mcp.get_saved_address()`. Mock: the `user_location` tool. Returns `{"status", "location", "delivery_area", "address_id"}`, or `{"status": "error", "message"}`. |

The `address_id` stored here is what the recommender and cart agent use for every Swiggy call.

## `__init__.py`
Re-exports `context_discovery_node`.
