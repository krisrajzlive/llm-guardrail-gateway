from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "chat"


@pytest.fixture(autouse=True)
def _local_prompts(monkeypatch):
    """Tests never call Bedrock Prompt Management: they read the repo's prompt files."""
    monkeypatch.setenv("PROMPT_SOURCE", "file")
    monkeypatch.setenv("PROMPT_DIR", str(PROMPT_DIR))
