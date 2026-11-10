"""AWS Lambda entrypoint (Function URL with AWS_IAM auth, or direct invoke)."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from .app import GatewayAgent, build_agent
from .models import ALIASES
from .sinks import stdout_sink

MAX_PROMPT_CHARS = 4000
_agent: GatewayAgent | None = None


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def _parse(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("body", event)
    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    global _agent
    try:
        body = _parse(event)
    except (ValueError, json.JSONDecodeError) as exc:
        return _response(400, {"error": f"invalid request: {exc}"})

    if body.get("health"):
        return _response(
            200,
            {
                "status": "ok",
                "models": list(ALIASES),
                "prompt_id": os.environ.get("PROMPT_ID"),
                "prompt_version": os.environ.get("PROMPT_VERSION"),
            },
        )

    prompt = body.get("prompt")
    alias = body.get("model", "claude")
    if not isinstance(prompt, str) or not prompt.strip():
        return _response(400, {"error": "'prompt' is required"})
    if len(prompt) > MAX_PROMPT_CHARS:
        return _response(400, {"error": f"'prompt' exceeds {MAX_PROMPT_CHARS} characters"})
    if alias not in ALIASES:
        return _response(400, {"error": f"'model' must be one of {list(ALIASES)}"})

    try:
        if _agent is None:
            _agent = build_agent(sink=stdout_sink)
        output = _agent.ask(prompt, alias)
    except Exception as exc:
        stdout_sink({"type": "request_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return _response(502, {"error": "upstream model call failed"})
    return _response(200, {"model": alias, "output": output})
