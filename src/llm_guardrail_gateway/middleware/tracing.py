"""Span-style tracing for every model call (latency, tokens, errors)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from ..sinks import Sink


def _model_name(model: Any) -> str:
    # Chat model classes expose their id under different attribute names (ChatBedrockConverse uses
    # model_id), so try the common ones and fall back to the class name.
    for attr in ("model_name", "model_id", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
    return type(model).__name__


def _last_ai_message(response: Any) -> AIMessage | None:
    # The handler may return a ModelResponse (with .result) or a bare message; handle both.
    messages = getattr(response, "result", None) or [response]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


class TracingMiddleware(AgentMiddleware):
    """Emit one ``model_call`` event per model attempt.

    Place it *inside* failover middleware so each attempt (including failed ones) is recorded.
    Independent of LangSmith: set ``LANGSMITH_TRACING=true`` to get native traces as well.
    """

    def __init__(self, sink: Sink) -> None:
        super().__init__()
        self._sink = sink

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        span: dict[str, Any] = {
            "type": "model_call",
            "span_id": uuid.uuid4().hex[:16],
            "model": _model_name(request.model),
            "n_messages": len(request.messages),
        }
        # perf_counter is monotonic, so latency is unaffected by system clock changes.
        start = time.perf_counter()
        try:
            response = handler(request)
        except Exception as exc:
            # Record the failure, then re-raise so the failover middleware (outside us) can react.
            span.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:300],  # truncated to keep log lines bounded
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            self._sink(span)
            raise
        last = _last_ai_message(response)
        usage: dict[str, Any] = dict(last.usage_metadata or {}) if last else {}
        span.update(
            status="ok",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            # Providers do not always report usage, so these can be None.
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        self._sink(span)
        return response
