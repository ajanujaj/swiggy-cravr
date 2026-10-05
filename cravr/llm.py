from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .debug_utils import debug

load_dotenv()

_llm: Any | None = None

DEFAULT_MODEL = "openai/gpt-oss-120b"


def _get_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_apikey")
    if not api_key:
        raise RuntimeError(
            "Missing Groq API key. Set GROQ_API_KEY (or groq_apikey) in your "
            ".env -- get a free key at https://console.groq.com/keys"
        )
    return api_key


def get_llm() -> Any:
    global _llm
    if _llm is None:
        try:
            from langchain_groq import ChatGroq
        except ImportError as error:
            raise RuntimeError(
                "Install requirements.txt to run Cravr's Groq-backed chat agent."
            ) from error
        model_name = os.getenv("CRAVR_MODEL", DEFAULT_MODEL)
        _llm = ChatGroq(model=model_name, temperature=0.3, api_key=_get_api_key())
        debug("llm", f"created ChatGroq model={model_name!r} temperature=0.3")
    return _llm
