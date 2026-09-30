# `cravr/` — the agent package

Cravr is a LangGraph chat agent that helps someone who is hungry but undecided pick food on Swiggy. Each user turn flows through a small graph of specialised nodes ("agents"), and all Swiggy access goes through MCP tools.

## How a turn flows

```mermaid
flowchart LR
    START --> R{_route_entry}
    R -- cart words --> CART[cart_agent]
    R -- reaction to shown dishes<br/>"2", "yes", "more" --> REC[recommendation_agent]
    R -- new request --> UP[user_preference]
    UP --> CTX[context_discovery] --> INT[intent_discovery] --> DEC[decision_agent] --> REC
    REC --> END
    CART --> END
```

1. **user_preference** loads saved preferences, order history and (on real Swiggy) the delivery address.
2. **context_discovery** fills in delivery context (location, area, `address_id`) without asking the user.
3. **intent_discovery** extracts what the user wants and asks follow-up questions (via `interrupt()`) until it knows enough.
4. **decision_agent** freezes everything into a `recommendation_brief`.
5. **recommendation_agent** searches, shows up to three dishes and handles the feedback loop. Picking a dish adds it to the cart.
6. **cart_agent** handles cart, coupon and checkout requests directly.

## Sub-packages

| Folder | Role |
|---|---|
| [`userPreference/`](userPreference/README.md) | Saved preferences + order history (first node) |
| [`context/`](context/README.md) | Delivery location / address context |
| [`intent/`](intent/README.md) | Understand the request; clarification questions |
| [`decision/`](decision/README.md) | Freeze the recommendation brief |
| [`recommend/`](recommend/README.md) | Search dishes, offer them, handle feedback |
| [`cart/`](cart/README.md) | Cart, add-ons/variants, coupons, checkout, payment |
| [`food_mcp/`](food_mcp/README.md) | Adapter over Swiggy's real Food MCP tools |

## Files in this folder

### `graph.py`
Builds and compiles the LangGraph workflow (with an in-memory checkpointer so interrupts can resume).

| Name | Usage |
|---|---|
| `_route_entry(state)` | Entry router. Returns `CART` if `is_cart_request`, `FEEDBACK` if `is_pure_feedback`, else `NEW_REQUEST`. |
| `build_graph()` | Registers the six nodes, wires the edges above and compiles with `MemorySaver`. |
| `app` | The compiled graph. Import this: `from cravr.graph import app`. |

### `state.py`
`CravrState` (TypedDict) is the shared state contract every node reads and writes: `messages` (chat history, merged by `add_messages`), `intent_profile`, `context_profile`, `preference_profile`, `history_profile`, `user_id`, `recommendation_brief`, `recommendation_state`, `cart_state`, `summary`, `decision`, `ready_to_recommend`.

### `client.py`
The one MCP connection helper every agent uses.

| Name | Usage |
|---|---|
| `DEFAULT_MCP_URL` / `MOCK_MCP_URL` | `CRAVR_MCP_URL` env var, defaulting to the local mock server. |
| `using_real_food_mcp()` | `True` when `CRAVR_MCP_URL` points at Swiggy's real Food MCP. Agents branch on this because tool names differ from the mock. |
| `_bearer_token()` | Reads the saved Swiggy OAuth token (see `mcp_server/oauth.py`) to send as `Authorization: Bearer`. |
| `call_tool(name, arguments, mcp_url=None)` | Async: open a `fastmcp.Client` and call one tool. |
| `call_tool_sync(name, arguments, mcp_url=None)` | Blocking wrapper used by the (sync) graph nodes. Don't call from inside a running event loop. |
| `main()` | Small manual smoke test. |

### `llm.py`
| Name | Usage |
|---|---|
| `get_llm()` | Lazily creates and caches a Groq `ChatGroq` model (`CRAVR_MODEL`, default `DEFAULT_MODEL`, temperature 0.3). |
| `_get_api_key()` | Reads `GROQ_API_KEY` (or `groq_apikey`); raises a helpful error if missing. |

### `schemas.py`
Pydantic models passed to `.with_structured_output(...)` so the LLM must return valid shapes.

| Class | Usage |
|---|---|
| `IntentExtraction` | Facts extracted from the user's message: `hunger_level`, `meal_intent`, `meal_time`, `craving`, `craving_tag`, `flavor`, `cuisine`. Used by `intent_discovery_node`. |
| `SuggestedOptions` | A short list of tappable answers for a clarification question. Used by `options.suggest_options`. |

### `options.py`
LLM-suggested multiple-choice answers for clarification questions.

| Name | Usage |
|---|---|
| `suggest_options(question, context=None, fallback=None, include_other=True, max_options=5)` | Asks the LLM for short answer choices and returns `[{"value","label"}]`, de-duplicated, ending with an "other" option. Never raises: falls back to `fallback`. |
| `resolve_option_choice(answer, options)` | Maps a reply (`"2"`, an option's value or label) back to that option's value. Unknown free text passes through unchanged. |
| `OTHER_VALUE` / `OTHER_LABEL` | The "type your own" escape hatch option. |

### `profile_utils.py`
| Name | Usage |
|---|---|
| `latest_user_text(messages)` | The most recent user message as plain text (works with dicts and LangChain message objects). |
| `merge_profile(previous, incoming)` | Layer newly extracted fields onto a running profile. Only non-empty values overwrite; lists are unioned. |
| `MANAGED_KEYS` | Computed keys (`missing_information`, `clarification_questions`, ...) that `merge_profile` never takes from the LLM. |

### `clock.py`
| Name | Usage |
|---|---|
| `IST` | Fixed UTC+05:30 timezone. |
| `now_ist()` | The app's single notion of "now". Servers run in UTC, so always use this for meal-time logic. |

### `debug_utils.py`
| Name | Usage |
|---|---|
| `debug_enabled()` | `True` when `CRAVR_DEBUG` is `1/true/yes/on`. |
| `debug(source, message)` | Prints `[debug:source] message` to stderr when debugging is on. |
| `test_mode_enabled()` | `True` when `CRAVR_TEST_MODE` is on. Checkout then fakes the order instead of calling Swiggy. |

### `run.py`
Command-line chat: `python -m cravr.run`.

| Name | Usage |
|---|---|
| `_load_app()` | Imports the compiled graph with a friendly error if dependencies are missing. |
| `_print_assistant_reply(result)` | Prints the last assistant message of a turn. |
| `_resume_until_complete(result, config, app)` | Prints each interrupt question with numbered options, reads your answer and resumes until the turn ends. |
| `main()` | The read-eval loop (Ctrl-C / Ctrl-D to exit). |

## Environment variables

| Variable | Effect |
|---|---|
| `GROQ_API_KEY` | Required for the LLM. |
| `CRAVR_MODEL` | Override the Groq model name. |
| `CRAVR_MCP_URL` | MCP server URL. Mock server by default; Swiggy's `https://mcp.swiggy.com/food` for real data. |
| `CRAVR_DEBUG` | Print per-node traces to stderr. |
| `CRAVR_TEST_MODE` | Fake checkout (no real order, no money). |
| `CRAVR_DISHES_FIRST` | `0` turns off the opening cards (usual / something new / right now) and goes straight to questions. |
| `CRAVR_WEATHER` | `0` turns off the Open-Meteo weather lookup. `CRAVR_WEATHER_CITY` overrides the city parsed from the address. |
| `CRAVR_PUBLIC_URL` | Base URL for group-order poll links (defaults to the `SWIGGY_REDIRECT_URI` host, then `http://localhost:8001`). |
| `DATABASE_URL` | Postgres URL (e.g. Render Postgres) for taste profiles and group polls, so they survive redeploys. `sqlite:///file.db` also works. Unset = JSON files. |
| `CRAVR_PREFS_PATH` / `CRAVR_GROUPS_PATH` | Where the learned taste profile and group polls are stored (JSON; defaults under `mcp_server/data/`). |

## Features added in the "undecided user" work

| Feature | Where |
|---|---|
| Dishes-first opening cards | `recommend/opening.py` |
| Ranking on history, delivery time, offers, open status | `recommend/signals.py`, `food_mcp/restaurants.py` |
| Learned taste profile (likes, dislikes, inferred diet, typical price) | `userPreference/taste.py` |
| Learning from "no" (asks why, fixes that, remembers it) | `recommend/feedback.py` |
| A reason on every card | `recommend/reasons.py` |
| Complete the meal (side + drink, same restaurant) | `cart/pairing.py` |
| Weather, mood and occasion themes | `context/weather.py`, `context/signals.py` |
| Group mode with a poll link | `group/` (poll page served by `group/web.py`) |
