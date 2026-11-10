"""Multi-provider model registry built on ``init_chat_model``."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Alias -> "provider:model". Note the Google provider key is ``google_genai``
# (underscore); the hyphenated form is not resolved by init_chat_model.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "openai": "openai:gpt-5.5",
    "anthropic": "anthropic:claude-opus-4-8",
    "google": "google_genai:gemini-3.1-pro-preview",
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
        _cache[alias] = init_chat_model(model_id(alias), max_retries=2, timeout=45)
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
