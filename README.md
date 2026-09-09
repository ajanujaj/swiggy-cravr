# Cravr

The active `cravr/` package is a minimal LangChain + LangGraph chat workflow. It stores conversation messages with LangGraph's `add_messages` reducer, calls Groq's hosted API in the chatbot node, and ends after one response.

## Run

Use Python 3.11 or 3.12 -- Chainlit's `gevent` dependency has no prebuilt wheel for 3.14+. On Anaconda: `conda create -n cravr python=3.11 -y && conda activate cravr`.

```bash
pip install -r requirements.txt
# set GROQ_API_KEY (or groq_apikey) in .env -- free key at https://console.groq.com/keys
python -m cravr.run
python -m unittest discover -s tests -v
CRAVR_DEBUG=1 python cravr/run.py  ## debug mode
```

Start the mock MCP server first (`python mcp_server/run_mcp.py`) -- `historyAgent`/`RecommendAgent` call it over HTTP.

## Run the chat UI

A [Chainlit](https://docs.chainlit.io) chat UI is available as an alternative to the CLI -- same backend, same graph.

```bash
pip install -r requirements.txt
# set GROQ_API_KEY (or groq_apikey) in .env
./run_mock_mcp.sh   # terminal 1, port 8000
./run_ui.sh         # terminal 2 -- chat UI + Swiggy OAuth callback, port 8001
```

Open **http://localhost:8001** (not 8000 -- that's the mock MCP server).

Use these two scripts rather than typing the commands by hand -- both default to port 8000, so a hand-typed command missing `--port 8001` silently collides with the mock server. Equivalent raw commands:

```bash
python mcp_server/run_mcp.py               # terminal 1, port 8000
uvicorn server:app --reload --port 8001    # terminal 2 -- --port is not optional
```

`server.py` is the entry point (not `UI/chainlit_app.py` directly) -- it mounts the Chainlit UI alongside the Swiggy OAuth callback routes on one process. To run the UI alone:

```bash
chainlit run UI/chainlit_app.py -w --port 8001
```

Set `CRAVR_DEBUG=1` in the server's own terminal to see `[debug:...]` traces:

```bash
CRAVR_DEBUG=1 uvicorn server:app --reload --port 8001
```

Environment variables:
- `CRAVR_MODEL` — Groq model id (must support tool calling). Default `openai/gpt-oss-120b`. See https://console.groq.com/docs/models.
- `GROQ_API_KEY` (or `groq_apikey`) — required, read from `.env`.
- `CRAVR_DEBUG=1` — print agent traces to stderr.
- `CRAVR_MCP_URL` — MCP server URL. Defaults to `http://127.0.0.1:8000/mcp` (mock); production should point at `https://mcp.swiggy.com/food`.

## Architecture

- **`cravr/`** — the LangGraph agent (intent, decision, recommendation) and the Groq LLM client. Framework-agnostic.
- **`UI/` + `server.py`** — the Chainlit chat UI and the Swiggy OAuth callback, deployed together as one service.
- **`mcp_server/`** — `run_mcp.py` is a standalone mock-data MCP tool server, standing in for Swiggy's real Food MCP; `oauth.py` is the real OAuth client and stays regardless of the mock server's fate. `cravr/client.py` reaches whichever MCP server is configured over HTTP via `CRAVR_MCP_URL`.

## Deploying

Both Render services need Python 3.12 -- a `.python-version` file at the repo root pins this (Render's own default hits the same Chainlit/gevent build failure as local dev on 3.14+).

Two services (see `render.yaml` -- deploy via Render's "New +" -> "Blueprint", or create by hand):

**`cravr-app`** — the chat UI + Swiggy OAuth callback, combined by `server.py`. This is the service whose URL matters to Swiggy.

```
build command: pip install -r requirements.txt
start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

**`cravr-mock-mcp`** — the standalone mock tool server.

```
build command: pip install -r requirements.txt
start command: python mcp_server/run_mcp.py
```

Set `cravr-app`'s `CRAVR_MCP_URL` to `cravr-mock-mcp`'s URL + `/mcp` (e.g. `https://cravr-mock-mcp.onrender.com/mcp`) while testing against mock data, or to `https://mcp.swiggy.com/food` once real access is live.

### Getting a URL for Swiggy to whitelist

Swiggy whitelists an OAuth **redirect URI**: `<cravr-app's real URL>/oauth/callback`. `server.py` serves that route directly (see `mcp_server/oauth.py`).

Render assigns `cravr-app`'s live URL from its name at creation time (may get a random suffix if taken) -- create the service first, copy the real URL, and submit `<that URL>/oauth/callback` to Swiggy. If it isn't `https://cravr-app.onrender.com`, also set `SWIGGY_REDIRECT_URI` to match.

Swiggy's Builders Club confirmed whitelisting the redirect URI is the entire setup needed -- no client_id, secret, or endpoint URLs issued manually. That's Dynamic Client Registration (RFC 7591): `/oauth/login` discovers Swiggy's OAuth endpoints via `.well-known` metadata (RFC 8414 / RFC 9728) and registers Cravr as a client on the fly, caching the result in `mcp_server/data/swiggy_client.json`.

Env vars, all optional (defaults are hardcoded per Swiggy's docs at https://mcp.swiggy.com/builders/docs/start/authenticate/):
- `SWIGGY_MCP_URL` — defaults to `https://mcp.swiggy.com`.
- `SWIGGY_REDIRECT_URI` — only if the Render URL isn't `https://cravr-app.onrender.com/oauth/callback`.
- `SWIGGY_SCOPES` — defaults to `mcp:tools mcp:resources mcp:prompts`.

Visiting `/oauth/login` discovers Swiggy's endpoints, registers Cravr (first time only), and runs the PKCE flow; `/oauth/callback` exchanges the code for a token, saved to `mcp_server/data/swiggy_tokens.json` (both files `.gitignore`d). Access tokens last 5 days; refresh tokens aren't issued in Swiggy's v1, so expiry means going through `/oauth/login` again.

Both Render free-tier services spin down when idle and cold-start on the next request (tens of seconds) -- not a bug, just the free tier.

## Swiggy Food MCP

Cravr targets only Swiggy's Food MCP server (`https://mcp.swiggy.com/food`) -- Instamart and Dineout are out of scope for now. Its tools, in the order a real order flow uses them (per https://mcp.swiggy.com/builders/docs/build/recipes/order-food/):

`get_addresses` -> `search_restaurants` -> `get_restaurant_menu` -> `update_food_cart` -> `fetch_food_coupons` / `apply_food_coupon` -> `get_food_cart` -> `place_food_order` -> `track_food_order` (plus `flush_food_cart`, `get_food_orders`). Cart is tied to one restaurant; max order value ₹1000; `place_food_order` is not idempotent.

`cravr/client.using_real_food_mcp()` is true once `CRAVR_MCP_URL` points at `mcp.swiggy.com`. `cravr/food_mcp.py` maps the real, read-only Food tools onto the same contracts the mock tools already provided:

- `fetch_user_location` -> `get_addresses`
- `_call_recommendation_tool` -> `search_restaurants` + `get_restaurant_menu`, flattened and filtered locally
- `fetch_order_history` -> `get_food_orders`

`fetch_user_preferences` stays on the mock server -- no Food MCP tool covers it.

Swiggy's docs don't publish exact response schemas, so `food_mcp.py`'s field-name guesses (`_first(...)` fallback lists) are unverified. Run with `CRAVR_DEBUG=1` the first time this hits the real server and check the `[debug:food_mcp] raw ... response` lines against what's actually returned; adjust the key lists if they differ. Cart/coupon/payment/`place_food_order`/`track_food_order` aren't wired up yet -- that's real-money territory and a separate phase.
