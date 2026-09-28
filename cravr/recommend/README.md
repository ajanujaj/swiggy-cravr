# `cravr/recommend/` — find dishes and handle feedback

Searches for dishes that match the frozen brief, shows them as a numbered list with tappable options, and reacts to the user's reply: pick, "more", "yes", "no", or a changed request. Picking a dish hands it straight to the cart.

## `RecommendAgent.py`

### Graph node

| Function | Usage |
|---|---|
| `recommendation_agent_node(state)` | **Graph node `recommendation_agent`.** If dishes are already on screen (`recommendation_state.awaiting_feedback`), interprets the reply: a pick → `_reply_with_pick`; "more" → next batch; "yes" with one dish → pick it; "no" → alternates or ask what to change; a single-dish decline → `widen_recommendations`. Otherwise runs a fresh `fetch_recommendations(brief)` and offers the results. |

### Public helpers (used by other packages)

| Function | Usage |
|---|---|
| `build_recommendation_brief(intent_profile, history_profile, preference_profile=None)` | Called once by `decision_agent`. Applies saved preferences, builds search filters and a history summary. Returns `{"filters", "history", "summary"}`. |
| `fetch_recommendations(brief, address_id=None, exclude_restaurant_ids=None)` | Searches with progressively looser filters (`_filter_variants`) until something matches, then ranks the results. `exclude_restaurant_ids` is used by the cart's "browse other restaurants". |
| `widen_recommendations(brief, address_id=None, exclude_items=None)` | A looser search for when only one dish matched and the user wants more options. |
| `is_pure_feedback(user_input, recommendation_state)` | Used by the graph's entry router. `True` when the message only reacts to dishes on screen ("3", "yes", "more"), so the turn skips intent/decision. Anything carrying a new constraint returns `False`. |

### Building the search

| Function | Usage |
|---|---|
| `_apply_saved_preferences(intent_profile, preference_profile)` | Fills in saved defaults (e.g. dietary type) only where this request didn't specify one. |
| `_build_recommendation_filters(intent_profile)` | Converts the intent profile into tool arguments: `cuisine`, `tags`, `dietary_type`, `max_price_inr`, `course`, `craving`. |
| `_course_preference(intent_profile)` | Which course is wanted (main / side / beverage / dessert); defaults to `main`. |
| `_summarise_history(history_profile)` | Favourite cuisines and repeat items from past orders. |
| `_brief_summary(filters, history)` | One-line human summary of the brief (for logs). |
| `_filter_variants(filters)` | Strict filters first, then looser ones. Dietary type and budget are never relaxed; cuisine is dropped before tags. |
| `_rank_candidates(recommendations, desired_tags, desired_cuisine, desired_course, history)` | Orders results by how well they match the original request (tags, cuisine, course, history), not by rating alone. |
| `_call_recommendation_tool(filters, address_id=None)` | Real Swiggy: `food_mcp.search_dishes`. Mock: the `recommendation` tool. |
| `_unwrap_tool_result(result)` | Pulls the dish list out of whatever shape the MCP result has. |
| `_resolve_address_id(address_id)` | Falls back to the saved address when none was passed. |

### Showing results and reading replies

| Function | Usage |
|---|---|
| `_describe(dish)` | "Dish from Restaurant (cuisine, ₹price, rating)". |
| `_build_recommendation_text(recommendations)` | The message text: a numbered list plus `PICK_PROMPT`, or a yes/no prompt for a single match. |
| `_numbered(dishes)` | The numbered lines. |
| `_dish_pick_options(dishes)` | Button options `{"value": "1", "label": dish name}`, matching the numbering. |
| `_resolve_choice(user_input, options)` | Which dish the user meant: "2", "the second one", "#3", or a dish name. Checked before yes/no. |
| `_looks_like_acceptance` / `_looks_like_rejection` / `_looks_like_more_request` | Whole-word checks against `ACCEPTANCE_TOKENS` / `REJECTION_TOKENS` / `MORE_TOKENS`. |
| `_looks_like_decline_only_offer(user_input)` | "No" / "widen" to a single-dish offer. |
| `_carries_new_constraints(user_input)` | `True` when the message changes the request ("make it veg"), not just reacts. |
| `_mentions(user_input, tokens)` | Whole-word/phrase matcher behind the checks above. |
| `_serve_next_batch(lead_in, offered, alternates, filters)` | Shows the next `DISPLAY_COUNT` (3) dishes from already-fetched results without searching again. |
| `_reply(content, **rec_state)` | Node return shape: one assistant message plus the updated `recommendation_state`. |
| `_reply_with_pick(chosen, address_id, state=None, **rec_state)` | The user picked a dish: adds it to the Swiggy cart via `cart.add_flow.add_dish_to_cart` and shows the cart. |

### `recommendation_state` keys
`last_recommendations` (dishes on screen), `alternates` (the rest of the result set), `filters`, `awaiting_feedback`, `satisfied`, `options` (buttons for the UI).

## `__init__.py`
Re-exports `recommendation_agent_node`, `build_recommendation_brief`, `fetch_recommendations`, `is_pure_feedback`.
