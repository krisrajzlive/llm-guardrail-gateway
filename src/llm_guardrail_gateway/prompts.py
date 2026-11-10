"""Prompt loading from Amazon Bedrock Prompt Management (production) or local files (offline/dev).

There are no prompt strings in application code. The text lives in ``prompts/chat/*.txt`` (reviewed in git),
is synced to a Bedrock managed prompt by ``scripts/prompt_registry.py``, and the Lambda reads an immutable,
numbered version of it at cold start.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def extract_variables(*templates: str) -> list[str]:
    return sorted({m for t in templates for m in _VAR.findall(t)})


def render(template: str, values: dict[str, str]) -> str:
    missing = [v for v in extract_variables(template) if v not in values]
    if missing:
        raise ValueError(f"missing prompt variables: {missing}")
    return _VAR.sub(lambda m: str(values[m.group(1)]), template)


@dataclass(frozen=True)
class ChatPrompt:
    """A system prompt plus a user-turn template; both may contain ``{{variable}}`` placeholders."""

    system: str
    user: str
    version: str = "local"

    @property
    def variables(self) -> list[str]:
        return extract_variables(self.system, self.user)

    def render_system(self, **values: str) -> str:
        return render(self.system, values)

    def render_user(self, **values: str) -> str:
        return render(self.user, values)


class PromptSource(Protocol):
    def load(self) -> ChatPrompt: ...


class FilePromptSource:
    """Reads ``system.txt`` and ``user.txt`` from a directory (tests, offline evals, local dev)."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def load(self) -> ChatPrompt:
        return ChatPrompt(
            system=(self._dir / "system.txt").read_text(encoding="utf-8").strip(),
            user=(self._dir / "user.txt").read_text(encoding="utf-8").strip(),
            version="file",
        )


def chat_prompt_from_response(response: dict[str, Any]) -> ChatPrompt:
    """Convert a bedrock-agent ``get_prompt`` response into a ChatPrompt."""
    variant = response["variants"][0]
    chat = variant["templateConfiguration"]["chat"]
    system = "\n".join(block["text"] for block in chat.get("system", []) if "text" in block)
    user = next(
        block["text"]
        for message in chat["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    )
    return ChatPrompt(system=system, user=user, version=str(response.get("version", "DRAFT")))


class BedrockPromptSource:
    """Reads a managed prompt (a numbered version in production, DRAFT in the eval gate)."""

    def __init__(self, prompt_id: str, version: str = "DRAFT", client: Any = None) -> None:
        self._prompt_id = prompt_id
        self._version = version
        self._client = client

    def load(self) -> ChatPrompt:
        client = self._client
        if client is None:
            import boto3  # provided by the Lambda runtime

            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            client = boto3.client("bedrock-agent", region_name=region)
        response = client.get_prompt(promptIdentifier=self._prompt_id, promptVersion=self._version)
        return chat_prompt_from_response(response)


def prompt_source_from_env() -> PromptSource:
    """Local: ``PROMPT_SOURCE=file`` (+ ``PROMPT_DIR``). Otherwise ``PROMPT_ID`` (+ ``PROMPT_VERSION``)."""
    if os.environ.get("PROMPT_SOURCE") == "file":
        return FilePromptSource(os.environ.get("PROMPT_DIR", "prompts/chat"))
    prompt_id = os.environ.get("PROMPT_ID")
    if not prompt_id:
        raise RuntimeError(
            "PROMPT_ID is not set. Run `python scripts/prompt_registry.py sync` and export the id, "
            "or set PROMPT_SOURCE=file for local/offline use."
        )
    return BedrockPromptSource(prompt_id, os.environ.get("PROMPT_VERSION", "DRAFT"))
