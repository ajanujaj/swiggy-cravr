"""The conversational LangChain node used by Cravr's LangGraph workflow."""

from __future__ import annotations

import os
from typing import Any

from .debug_utils import debug

_llm: Any | None = None

# llama3.2:1b is too small to reliably follow structured-output/JSON
# instructions (that's what caused the "predefined fallback response"
# behavior). Default to a 7B instruction-tuned model that supports Ollama's
# tool-calling, which `.with_structured_output()` relies on. Override with
# `CRAVR_MODEL=<ollama-tag>` for anything else you've already pulled -- it
# must support tool calling for structured output to work.
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def get_llm() -> Any:
    """Create the local Ollama-backed chat model only when the graph runs."""
    global _llm
    if _llm is None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Install requirements.txt to run Cravr's local Llama chat agent.") from error
        model_name = os.getenv("CRAVR_MODEL", DEFAULT_MODEL)
        _llm = ChatOllama(model=model_name, temperature=0.3)
        debug("llm", f"created ChatOllama model={model_name!r} temperature=0.3")
    return _llm
