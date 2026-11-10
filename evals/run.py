"""Offline/live eval runner and CI gate.

    python -m evals.run                                  # live: 3 Bedrock models, evals/thresholds.json
    python -m evals.run --models claude --min-score 0.8
    python -m evals.run --offline --min-score 1.0        # deterministic harness check, no keys

Exit code is 1 if any model scores below ``--min-score`` (this is the deploy gate).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from llm_guardrail_gateway.app import build_agent
from llm_guardrail_gateway.models import ALIASES, override_models
from llm_guardrail_gateway.prompts import ChatPrompt, FilePromptSource, prompt_source_from_env
from llm_guardrail_gateway.sinks import ListSink
from llm_guardrail_gateway.testing import ScriptedChatModel

from .scoring import score_case

DATASET = Path(__file__).with_name("dataset.json")
THRESHOLDS = Path(__file__).with_name("thresholds.json")
RESULTS_DIR = Path(__file__).with_name("results")


def run_model(alias: str, cases: list[dict[str, Any]], prompt: ChatPrompt) -> dict[str, Any]:
    sink = ListSink()
    # No failover noise in evals: a provider outage should fail the gate, not be masked.
    agent = build_agent(prompt=prompt, sink=sink, default_alias=alias, enforce_inline_evals=False)
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            output = agent.ask(case["prompt"], alias)
            error = None
        except Exception as exc:
            output, error = "", f"{type(exc).__name__}: {exc}"[:200]
        passed, failures = (False, [error]) if error else score_case(output, case["expect"])
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "failures": failures,
                "latency_s": round(time.perf_counter() - started, 2),
                "output": output[:300],
            }
        )
    failovers = len(sink.of_type("failover"))
    score = sum(r["passed"] for r in rows) / len(rows)
    return {
        "model": alias,
        "score": round(score, 3),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "failovers": failovers,
        "cases": rows,
    }


def resolve_thresholds(models: list[str], override: float | None) -> dict[str, float]:
    """One value for every model if ``override`` is given, else evals/thresholds.json."""
    if override is not None:
        return dict.fromkeys(models, override)
    config = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    return {m: float(config.get(m, config["default"])) for m in models}


def render_table(results: list[dict[str, Any]], thresholds: dict[str, float]) -> str:
    lines = [
        "| model | score | min | p50 latency (s) | failed cases | gate |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        failed = ", ".join(c["id"] for c in r["cases"] if not c["passed"]) or "-"
        minimum = thresholds[r["model"]]
        gate = "PASS" if r["score"] >= minimum else "FAIL"
        lines.append(
            f"| {r['model']} | {r['score']:.2f} | {minimum:.2f} | {r['p50_latency_s']} | {failed} | {gate} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(ALIASES), choices=ALIASES)
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.environ["EVAL_MIN_SCORE"]) if os.environ.get("EVAL_MIN_SCORE") else None,
        help="override: one minimum for every model (default: evals/thresholds.json, or $EVAL_MIN_SCORE)",
    )
    parser.add_argument("--offline", action="store_true", help="use scripted fake models")
    args = parser.parse_args(argv)

    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    # Offline: the repo's prompt files. Live: the Bedrock managed prompt (PROMPT_ID, version DRAFT unless
    # PROMPT_VERSION is set), i.e. exactly the prompt about to be deployed.
    source = FilePromptSource("prompts/chat") if args.offline else prompt_source_from_env()
    prompt = source.load()
    print(f"prompt version under test: {prompt.version}\n")
    with ExitStack() as stack:
        if args.offline:
            answers = {c["prompt"]: c["offline_answer"] for c in cases}
            stack.enter_context(
                override_models({a: ScriptedChatModel(answers=answers, model_name=a) for a in ALIASES})
            )
        results = [run_model(alias, cases, prompt) for alias in args.models]

    thresholds = resolve_thresholds(args.models, args.min_score)
    table = render_table(results, thresholds)
    print(table)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Eval gate\n\n{table}\n")

    failing = [r["model"] for r in results if r["score"] < thresholds[r["model"]]]
    if failing:
        print(f"\nGATE FAILED for: {', '.join(failing)}", file=sys.stderr)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
