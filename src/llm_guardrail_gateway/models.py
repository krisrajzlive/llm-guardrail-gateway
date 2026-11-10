"""Multi-provider model registry built on ``init_chat_model``."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Alias -> "provider:model". Every alias runs on Amazon Bedrock (Converse API) so no third-party API
# keys are needed: auth is the standard AWS credential chain (IAM role on Lambda / in CI, profile locally).
# Cross-region inference profiles (``us.`` prefix) are used for the lightweight models.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "claude": "bedrock_converse:us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "nova": "bedrock_converse:us.amazon.nova-lite-v1:0",
    "llama": "bedrock_converse:us.meta.llama3-1-8b-instruct-v1:0",
}
ALIASES: tuple[str, ...] = tuple(DEFAULT_MODEL_IDS)

_cache: dict[str, BaseChatModel] = {}
_overrides: dict[str, BaseChatModel] = {}


def model_id(alias: str) -> str:
    """Resolve an alias to a ``provider:model`` string (env override: LLM_MODEL_<ALIAS>)."""
    if alias not in DEFAULT_MODEL_IDS:
        raise ValueError(f"Unknown model alias {alias!r}; expected one of {list(ALIASES)}")
    return os.environ.get(f"LLM_MODEL_{alias.upper()}", DEFAULT_MODEL_IDS[alias])


def get_model(alias: str) -> BaseChatModel:
    """Return a cached chat model for the alias (initialised lazily, so missing keys fail late)."""
    if alias in _overrides:
        return _overrides[alias]
    if alias not in _cache:
        _cache[alias] = init_chat_model(
            model_id(alias),
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            temperature=0,
            max_tokens=1024,
        )
    return _cache[alias]


@contextmanager
def override_models(models: Mapping[str, BaseChatModel]) -> Iterator[None]:
    """Temporarily replace aliases with concrete models (tests and offline evals)."""
    previous = dict(_overrides)
    _overrides.update(models)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(previous)
