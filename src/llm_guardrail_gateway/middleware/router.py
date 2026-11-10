"""Per-request model selection via runtime context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from ..models import get_model


@dataclass
class GatewayContext:
    """Per-invocation context: which provider alias should serve this request."""

    model: str = "claude"


class ModelRouterMiddleware(AgentMiddleware):
    """Swap the request's model for the alias named in ``context.model``."""

    def __init__(self, default_alias: str = "claude") -> None:
        super().__init__()
        self._default = default_alias

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        # Context is optional in LangChain, so fall back to the default alias if none was supplied.
        ctx = request.runtime.context
        alias = getattr(ctx, "model", None) or self._default
        # request.override() returns a modified copy; the original request is never mutated.
        return handler(request.override(model=get_model(alias)))
