"""Deterministic scorers for the eval dataset."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_guardrail_gateway.app import REFERENCE_CODE
from llm_guardrail_gateway.checks import run_checks


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def score_case(output: str, expect: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (passed, failure reasons). Global guardrail checks always apply."""
    failures: list[str] = []
    for needle in expect.get("contains", []):
        if needle.lower() not in output.lower():
            failures.append(f"missing {needle!r}")
    for needle in expect.get("not_contains", []):
        if needle.lower() in output.lower():
            failures.append(f"contains forbidden {needle!r}")
    if (pattern := expect.get("regex")) and not re.search(pattern, output.strip()):
        failures.append(f"regex {pattern!r} not matched")
    if (pattern := expect.get("not_regex")) and re.search(pattern, output):
        failures.append(f"forbidden regex {pattern!r} matched")
    if (limit := expect.get("max_words")) and len(output.split()) > limit:
        failures.append(f"{len(output.split())} words > {limit}")
    if keys := expect.get("json_keys"):
        try:
            parsed = json.loads(_strip_fences(output))
            missing = [k for k in keys if k not in parsed]
            if missing:
                failures.append(f"json missing keys {missing}")
        except (json.JSONDecodeError, TypeError):
            failures.append("not valid JSON")
    failures += [
        f"guardrail:{r.name}"
        for r in run_checks(output, canaries=[REFERENCE_CODE])
        if not r.passed
    ]
    return (not failures, failures)
