#!/usr/bin/env bash
# Terminal 1. Starts the mock MCP tool server on port 8000 -- fixed, not
# configurable via a flag here, so it can't drift out of sync with
# CRAVR_MCP_URL's documented default (http://127.0.0.1:8000/mcp).
set -e
python mcp_server/run_mcp.py
