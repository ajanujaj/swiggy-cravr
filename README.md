# Cravr

The active `cravr/` package is a minimal LangChain + LangGraph chat workflow. It stores conversation messages with LangGraph's `add_messages` reducer, calls a local Ollama Llama model in the chatbot node, and ends after one response.

## Run

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct   # default model; must support tool calling for structured output
python -m cravr.run
python -m unittest discover -s tests -v
CRAVR_DEBUG=1 python cravr/run.py  ## debug model
```

Also start the MCP server (`python mcp_server/run_mcp.py`) before running the CLI, since `historyAgent`/`RecommendAgent` call it over HTTP.

Environment variables:
- `CRAVR_MODEL` — override the Ollama model tag (must support tool calling, e.g. `llama3.1:8b`, `qwen2.5:7b-instruct`; `llama3.2:1b` does not).
- `CRAVR_DEBUG=1` — print each agent's raw LLM output, merged profile, and routing decisions to stderr.
- `CRAVR_MCP_URL` — override the MCP server URL (defaults to `http://127.0.0.1:8000/mcp`).


