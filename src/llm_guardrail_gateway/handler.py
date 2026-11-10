"""AWS Lambda entrypoint (Function URL with AWS_IAM auth, or direct invoke)."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from .app import GatewayAgent, build_agent
from .models import ALIASES
from .sinks import stdout_sink

# Bounds cost and abuse: the function URL is IAM-protected, but a caller could still send huge prompts.
MAX_PROMPT_CHARS = 4000
# Module-level so a warm container builds the agent (and loads the prompt) once, not per request.
_agent: GatewayAgent | None = None


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Function-URL-shaped response; direct invokes get the same dict back."""
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def _parse(event: dict[str, Any]) -> dict[str, Any]:
    # A Function URL wraps the request in {"body": "<json string>"}; a console/CLI test event is the
    # plain JSON object itself. Accept both so the same code serves every way of calling it.
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

    # Health check: no model call, so it is free and fast. It also reports the pinned prompt version,
    # which the deploy smoke test uses to confirm the new version actually went live.
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

    # Validate everything before spending a model call.
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
        # Log the detail for operators, but return a generic message so internals never reach callers.
        stdout_sink({"type": "request_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return _response(502, {"error": "upstream model call failed"})
    return _response(200, {"model": alias, "output": output})
