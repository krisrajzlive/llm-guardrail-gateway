"""Agent assembly. Prompts come from Bedrock Prompt Management (see ``prompts.py``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent

from .middleware import (
    GatewayContext,
    InlineEvalMiddleware,
    ModelRouterMiddleware,
    ProviderFailoverMiddleware,
    TracingMiddleware,
)
from .models import get_model
from .prompts import ChatPrompt, prompt_source_from_env
from .sinks import Sink, stdout_sink

# Deployment-specific secret planted in the managed system prompt to detect prompt leakage.
# Configurable (REFERENCE_CODE); the value is not part of any prompt text.
REFERENCE_CODE = os.environ.get("REFERENCE_CODE", "CANARY-7f3a91")


@dataclass
class GatewayAgent:
    """A LangChain agent plus the managed prompt it was built from."""

    agent: Any
    prompt: ChatPrompt

    def ask(self, question: str, model: str = "claude") -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": self.prompt.render_user(question=question)}]},
            context=GatewayContext(model=model),
        )
        return str(result["messages"][-1].text)


def build_agent(
    *,
    prompt: ChatPrompt | None = None,
    sink: Sink = stdout_sink,
    default_alias: str = "claude",
    enforce_inline_evals: bool = True,
    tools: list[Any] | None = None,
) -> GatewayAgent:
    """Build the agent.

    ``prompt`` defaults to the prompt source named by the environment (Bedrock managed prompt in
    production). Middleware order matters (first = outermost): route -> failover -> trace, so every
    attempt, including failed ones, is traced, and inline evals see only the final response.
    """
    prompt = prompt or prompt_source_from_env().load()
    agent = create_agent(  # type: ignore[misc]
        model=get_model(default_alias),
        tools=tools or [],
        system_prompt=prompt.render_system(reference_code=REFERENCE_CODE),
        middleware=[
            ModelRouterMiddleware(default_alias),
            ProviderFailoverMiddleware(sink),
            TracingMiddleware(sink),
            InlineEvalMiddleware(sink, canaries=[REFERENCE_CODE], enforce=enforce_inline_evals),
        ],
        context_schema=GatewayContext,
    )
    return GatewayAgent(agent=agent, prompt=prompt)
