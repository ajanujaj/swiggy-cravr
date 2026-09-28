# `cravr/userPreference/` — preferences & order history

The **first node** of a new request. Loads what we already know about the user before asking anything.

## `preference.py`

| Function | Usage |
|---|---|
| `user_preference_node(state)` | **Graph node `user_preference`.** On real Swiggy, if no `address_id` is known yet, resolves the saved address first (via `food_mcp.get_saved_address`). Then loads saved preferences and order history. Returns `preference_profile`, `history_profile`, `user_id`, and an updated `context_profile` when an address was found. |
| `fetch_user_preferences(user_id)` | Calls the mock `user_preferences` tool. Always uses `MOCK_MCP_URL`, because Swiggy's Food MCP has no preferences tool. |
| `fetch_order_history(address_id=None, *, user_id=None)` | Real Swiggy: `food_mcp.get_order_history(address_id)`. Mock: the `order_history` tool with `user_id`. Returns `{"status", "count", "orders"}`. |
| `history_discovery_node(state)` | Older stand-alone node that only fetches order history. Exported but not wired into `graph.py`; `user_preference_node` does this now. |
| `_call_mcp_tool(tool_name, user_id, mcp_url=None)` | Calls one user-scoped MCP tool and normalises the FastMCP response into a dict. |

## `__init__.py`
Re-exports `user_preference_node`, `history_discovery_node`, `fetch_order_history`, `fetch_user_preferences`.
