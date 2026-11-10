"""Deterministic chat model for tests and offline eval runs (no network, no keys)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Answers from a prompt->answer table; can be told to fail."""

    answers: dict[str, str] = {}
    default: str = "ok"
    fail_with: str | None = None
    model_name: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        prompt = human.text if human else ""
        # Match on the longest known question contained in the (possibly templated) user turn, so the
        # answer does not depend on how the managed prompt wraps the question.
        matches = [key for key in self.answers if key in prompt]
        text = self.answers[max(matches, key=len)] if matches else self.default
        n_in, n_out = len(prompt.split()), len(text.split())
        message = AIMessage(
            content=text,
            usage_metadata={"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
