#!/usr/bin/env bash
# Terminal 2. Starts the chat UI + Swiggy OAuth callback on port 8001 --
# deliberately NOT 8000, which run_mock_mcp.sh (terminal 1) already owns.
# Open http://localhost:8001 once this is running, not 8000.
set -e
uvicorn server:app --reload --port 8001
