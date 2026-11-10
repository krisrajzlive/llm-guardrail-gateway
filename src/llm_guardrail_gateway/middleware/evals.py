"""Inline (online) evals: run cheap checks on every model response."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..checks import all_passed, run_checks
from ..sinks import Sink

BLOCKED_MESSAGE = "I can't provide that response."


class InlineEvalMiddleware(AgentMiddleware):
    """Score each final model response and emit an ``inline_eval`` event.

    With ``enforce=True`` a failing response is replaced by a safe refusal (same message id,
    so the add_messages reducer overwrites it).
    """

    def __init__(self, sink: Sink, *, canaries: Iterable[str] = (), enforce: bool = False) -> None:
        super().__init__()
        self._sink = sink
        self._canaries = tuple(canaries)
        self._enforce = enforce

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        results = run_checks(last.text, canaries=self._canaries)
        passed = all_passed(results)
        self._sink(
            {
                "type": "inline_eval",
                "passed": passed,
                "failed_checks": [r.name for r in results if not r.passed],
                "enforced": self._enforce and not passed,
            }
        )
        if self._enforce and not passed:
            return {"messages": [AIMessage(content=BLOCKED_MESSAGE, id=last.id)]}
        return None
