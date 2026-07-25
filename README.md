# Cravr

The active `cravr/` package is a minimal LangChain + LangGraph chat workflow. It stores conversation messages with LangGraph's `add_messages` reducer, calls a local Ollama Llama model in the chatbot node, and ends after one response.

## Run

```bash
pip install -r requirements.txt
ollama pull llama3.2:1b
python -m cravr.run
python -m unittest discover -s tests -v
```

The former multi-agent prototype is preserved in `cravrbacklog/`.


