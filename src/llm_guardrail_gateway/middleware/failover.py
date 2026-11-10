"""Fail over to another provider when the active one errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from ..models import ALIASES, get_model
from ..sinks import Sink


class ProviderFailoverMiddleware(AgentMiddleware):
    """Try the request's model; on any exception try the other aliases in order.

    Unlike the built-in ``ModelFallbackMiddleware`` (a static list), the alternatives here are
    derived from whichever alias the router picked, so the active provider is never retried.
    """

    def __init__(self, sink: Sink, order: Sequence[str] = ALIASES) -> None:
        super().__init__()
        self._sink = sink
        self._order = tuple(order)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        try:
            # Happy path: the model the router chose just answers.
            return handler(request)
        except Exception as first_error:
            active = getattr(request.runtime.context, "model", None)
            last_error: Exception = first_error
            # Skip the alias that just failed; retrying it would only repeat the same error.
            for alias in (a for a in self._order if a != active):
                try:
                    # get_model() is inside the try on purpose: a backup model that cannot even be
                    # created (no access, bad id) should just be skipped, not abort the failover.
                    response = handler(request.override(model=get_model(alias)))
                except Exception as exc:
                    last_error = exc
                    continue
                # Emit only after a backup succeeds, so the event always means "failover worked".
                self._sink(
                    {
                        "type": "failover",
                        "from": active,
                        "to": alias,
                        "reason": f"{type(first_error).__name__}: {first_error}"[:300],
                    }
                )
                return response
            # Every model failed: surface the last error, chained to the original for debugging.
            raise last_error from first_error
