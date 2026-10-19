"""LLM Client Factory & Structured Output Invocation Module.

Wraps ChatGroq with:
- Configurable model selection via LLM_MODEL environment variable
- Deterministic temperature=0 settings
- Configurable timeouts and exponential-backoff retries
- Graceful exception interception returning None for zero-downtime offline rule fallback
"""

import logging
import os
import time
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("triage-llm")

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def get_llm_model_name() -> str:
    """Return the active LLM model identifier."""
    return os.environ.get("LLM_MODEL") or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def is_llm_available() -> bool:
    """Check whether a valid Groq API key is present."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return bool(key) and not key.startswith("your_")


def get_llm_client(
    temperature: float = 0.0,
    timeout: float = 5.0,
    max_retries: int = 2,
    model_name: Optional[str] = None,
) -> Any:
    """Instantiate and return a ChatGroq client or None if unconfigured."""
    if not is_llm_available():
        return None

    try:
        from langchain_groq import ChatGroq

        model = model_name or get_llm_model_name()
        return ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:
        logger.warning("Failed to initialize ChatGroq client: %s", exc)
        return None


def call_structured_llm(
    schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str] = None,
    timeout: float = 5.0,
    max_retries: int = 1,
) -> Optional[T]:
    """Execute a structured JSON-mode LLM call returning a validated Pydantic model instance.

    On any failure (network error, rate limit HTTP 429, schema validation error), returns None.
    """
    llm = get_llm_client(timeout=timeout, max_retries=max_retries)
    if llm is None:
        return None

    try:
        from langchain_core.prompts import ChatPromptTemplate

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", "{input}"))

        prompt_tpl = ChatPromptTemplate.from_messages(messages)
        structured_llm = llm.with_structured_output(schema)
        chain = prompt_tpl | structured_llm

        t0 = time.perf_counter()
        result = chain.invoke({"input": user_prompt})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("Structured LLM call succeeded in %.1f ms", latency_ms)
        return result
    except Exception as exc:
        logger.warning("Structured LLM invocation failed, falling back to rules: %s", exc)
        return None
