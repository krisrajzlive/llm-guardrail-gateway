"""Sync the git-tracked prompt to Bedrock Prompt Management and publish immutable versions.

    python scripts/prompt_registry.py sync [--github-output]
        Create or update the managed prompt's DRAFT from prompts/chat/. The eval gate tests DRAFT.

    python scripts/prompt_registry.py publish --prompt-id ID [--github-output]
        Snapshot DRAFT as a numbered version (skipped if it equals the latest version) and print it.
        The Lambda is pinned to this number, so prompt changes only reach production after the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from llm_guardrail_gateway.models import DEFAULT_MODEL_IDS
from llm_guardrail_gateway.prompts import (
    ChatPrompt,
    FilePromptSource,
    chat_prompt_from_response,
)

PROMPT_NAME = "llm-guardrail-gateway-chat"
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "chat"


def _client() -> Any:
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("bedrock-agent", region_name=region)


def build_variant(prompt: ChatPrompt) -> dict[str, Any]:
    model = DEFAULT_MODEL_IDS["claude"].split(":", 1)[1]
    return {
        "name": "default",
        "templateType": "CHAT",
        "modelId": model,
        "templateConfiguration": {
            "chat": {
                "system": [{"text": prompt.system}],
                "messages": [{"role": "user", "content": [{"text": prompt.user}]}],
                "inputVariables": [{"name": v} for v in prompt.variables],
            }
        },
    }


def _list_summaries(client: Any, **kwargs: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        page = client.list_prompts(**kwargs, **({"nextToken": token} if token else {}))
        items += page.get("promptSummaries", [])
        token = page.get("nextToken")
        if not token:
            return items


def find_prompt_id(client: Any, name: str = PROMPT_NAME) -> str | None:
    for summary in _list_summaries(client):
        if summary["name"] == name:
            return str(summary["id"])
    return None


def sync(client: Any, prompt: ChatPrompt) -> dict[str, str]:
    variants = [build_variant(prompt)]
    prompt_id = find_prompt_id(client)
    if prompt_id is None:
        created = client.create_prompt(
            name=PROMPT_NAME,
            description="System and user-turn prompt for llm-guardrail-gateway (source: prompts/chat in git)",
            variants=variants,
            defaultVariant="default",
        )
        prompt_id, action = created["id"], "created"
    else:
        client.update_prompt(
            promptIdentifier=prompt_id,
            name=PROMPT_NAME,
            variants=variants,
            defaultVariant="default",
        )
        action = "updated"
    return {"prompt_id": prompt_id, "action": action}


def _same(a: ChatPrompt, b: ChatPrompt) -> bool:
    return (a.system, a.user) == (b.system, b.user)


def publish(client: Any, prompt_id: str) -> dict[str, str]:
    draft = chat_prompt_from_response(client.get_prompt(promptIdentifier=prompt_id))
    numbered = [
        s["version"] for s in _list_summaries(client, promptIdentifier=prompt_id) if s["version"] != "DRAFT"
    ]
    if numbered:
        latest = max(numbered, key=int)
        latest_prompt = chat_prompt_from_response(
            client.get_prompt(promptIdentifier=prompt_id, promptVersion=latest)
        )
        if _same(draft, latest_prompt):
            return {"prompt_id": prompt_id, "prompt_version": str(latest), "action": "reused"}
    version = client.create_prompt_version(promptIdentifier=prompt_id)["version"]
    return {"prompt_id": prompt_id, "prompt_version": str(version), "action": "created"}


def _emit(result: dict[str, str], github_output: bool) -> None:
    print(json.dumps(result))
    if github_output and (path := os.environ.get("GITHUB_OUTPUT")):
        with open(path, "a", encoding="utf-8") as fh:
            for key, value in result.items():
                fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None, client: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "publish"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--github-output", action="store_true")
        if name == "publish":
            cmd.add_argument("--prompt-id", required=True)
    args = parser.parse_args(argv)
    client = client or _client()
    if args.command == "sync":
        _emit(sync(client, FilePromptSource(PROMPT_DIR).load()), args.github_output)
    else:
        _emit(publish(client, args.prompt_id), args.github_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
