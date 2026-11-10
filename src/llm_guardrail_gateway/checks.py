"""Cheap deterministic output checks, shared by inline middleware and offline evals."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def find_pii(text: str) -> list[str]:
    hits = [f"email:{m}" for m in _EMAIL.findall(text)]
    hits += [f"ssn:{m}" for m in _SSN.findall(text)]
    hits += [f"card:{m.strip()}" for m in _CARD.findall(text) if _luhn(m)]
    return hits


def run_checks(
    text: str, *, max_chars: int = 4000, canaries: Iterable[str] = ()
) -> list[CheckResult]:
    pii = find_pii(text)
    leaked = [c for c in canaries if c and c in text]
    return [
        CheckResult("non_empty", bool(text.strip())),
        CheckResult("within_length", len(text) <= max_chars, f"{len(text)} chars"),
        CheckResult("no_pii", not pii, ", ".join(pii)),
        CheckResult("no_secrets", not _SECRET.search(text)),
        CheckResult("no_canary_leak", not leaked, ", ".join(leaked)),
    ]


def all_passed(results: Iterable[CheckResult]) -> bool:
    return all(r.passed for r in results)
