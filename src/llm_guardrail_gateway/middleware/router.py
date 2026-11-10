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

    model: str = "anthropic"


class ModelRouterMiddleware(AgentMiddleware):
    """Swap the request's model for the alias named in ``context.model``."""

    def __init__(self, default_alias: str = "anthropic") -> None:
        super().__init__()
        self._default = default_alias

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        ctx = request.runtime.context
        alias = getattr(ctx, "model", None) or self._default
        return handler(request.override(model=get_model(alias)))
