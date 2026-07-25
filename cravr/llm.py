"""The conversational LangChain node used by Cravr's LangGraph workflow."""

from __future__ import annotations

from typing import Any

_llm: Any | None = None

def get_llm() -> Any:
    """Create the local Ollama-backed Llama model only when the graph runs."""
    global _llm
    if _llm is None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Install requirements.txt to run Cravr's local Llama chat agent.") from error
        _llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
    return _llm
