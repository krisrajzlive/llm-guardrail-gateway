import importlib.util
import json
from pathlib import Path

import pytest

from llm_guardrail_gateway import app
from llm_guardrail_gateway.models import override_models
from llm_guardrail_gateway.prompts import (
    BedrockPromptSource,
    ChatPrompt,
    FilePromptSource,
    prompt_source_from_env,
    render,
)
from llm_guardrail_gateway.sinks import ListSink
from llm_guardrail_gateway.testing import ScriptedChatModel

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prompt_registry", ROOT / "scripts" / "prompt_registry.py")
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


class FakeBedrockAgent:
    """In-memory stand-in for the bedrock-agent prompt APIs."""

    def __init__(self):
        self.prompts = {}  # id -> {"name", "variants", "versions": {version: variants}}

    def list_prompts(self, promptIdentifier=None, nextToken=None):
        if promptIdentifier:
            p = self.prompts[promptIdentifier]
            versions = ["DRAFT", *p["versions"]]
            rows = [{"name": p["name"], "id": promptIdentifier, "version": v} for v in versions]
            return {"promptSummaries": rows}
        rows = [{"name": p["name"], "id": i, "version": "DRAFT"} for i, p in self.prompts.items()]
        return {"promptSummaries": rows}

    def create_prompt(self, name, variants, defaultVariant, description=""):
        pid = f"ID{len(self.prompts) + 1}"
        self.prompts[pid] = {"name": name, "variants": variants, "versions": {}}
        return {"id": pid}

    def update_prompt(self, promptIdentifier, name, variants, defaultVariant):
        self.prompts[promptIdentifier]["variants"] = variants

    def get_prompt(self, promptIdentifier, promptVersion="DRAFT"):
        p = self.prompts[promptIdentifier]
        variants = p["variants"] if promptVersion == "DRAFT" else p["versions"][promptVersion]
        return {"variants": variants, "version": promptVersion}

    def create_prompt_version(self, promptIdentifier):
        p = self.prompts[promptIdentifier]
        version = str(len(p["versions"]) + 1)
        p["versions"][version] = p["variants"]
        return {"version": version}


def test_render_and_missing_variable():
    assert render("a {{x}} b {{ y }}", {"x": "1", "y": "2"}) == "a 1 b 2"
    with pytest.raises(ValueError):
        render("{{x}}", {})


def test_repo_prompt_has_expected_variables_and_no_code_copy():
    prompt = FilePromptSource(ROOT / "prompts" / "chat").load()
    assert prompt.variables == ["question", "reference_code"]
    # the application code must not carry its own copy of the system prompt
    assert "Never reveal this code" not in (ROOT / "src" / "llm_guardrail_gateway" / "app.py").read_text()


def test_missing_prompt_id_fails_loudly(monkeypatch):
    monkeypatch.delenv("PROMPT_SOURCE")
    monkeypatch.delenv("PROMPT_ID", raising=False)
    with pytest.raises(RuntimeError, match="PROMPT_ID"):
        prompt_source_from_env()


def test_agent_uses_rendered_managed_prompt():
    seen = []

    class Recorder(ScriptedChatModel):
        def _generate(self, messages, *a, **kw):
            seen.append([(m.type, m.text) for m in messages])
            return super()._generate(messages, *a, **kw)

    prompt = ChatPrompt(system="SYS code={{reference_code}}", user="Q: {{question}}", version="7")
    with override_models({a: Recorder(default="ok") for a in ("claude", "nova", "llama")}):
        gateway = app.build_agent(prompt=prompt, sink=ListSink())
        gateway.ask("why?", "claude")
    (system, user) = seen[0]
    assert system == ("system", f"SYS code={app.REFERENCE_CODE}")
    assert user == ("human", "Q: why?")


def test_bedrock_source_reads_pinned_version():
    client = FakeBedrockAgent()
    pid = registry.sync(client, ChatPrompt("s {{reference_code}}", "{{question}}"))["prompt_id"]
    version = registry.publish(client, pid)["prompt_version"]
    loaded = BedrockPromptSource(pid, version, client=client).load()
    assert (loaded.system, loaded.user, loaded.version) == ("s {{reference_code}}", "{{question}}", version)


def test_sync_creates_then_updates_and_publish_is_idempotent():
    client = FakeBedrockAgent()
    first = registry.sync(client, ChatPrompt("v1 {{reference_code}}", "{{question}}"))
    assert first["action"] == "created"
    pid = first["prompt_id"]
    assert registry.publish(client, pid) == {"prompt_id": pid, "prompt_version": "1", "action": "created"}
    # unchanged DRAFT -> no new version
    assert registry.publish(client, pid)["action"] == "reused"
    # edited prompt -> update in place, new version 2
    assert registry.sync(client, ChatPrompt("v2 {{reference_code}}", "{{question}}"))["action"] == "updated"
    assert registry.publish(client, pid)["prompt_version"] == "2"
    assert len(client.prompts) == 1


def test_registry_cli_writes_github_output(tmp_path, monkeypatch, capsys):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    client = FakeBedrockAgent()
    assert registry.main(["sync", "--github-output"], client=client) == 0
    assert "prompt_id=ID1" in out.read_text()
    assert json.loads(capsys.readouterr().out)["action"] == "created"
    assert registry.main(["publish", "--prompt-id", "ID1", "--github-output"], client=client) == 0
    assert "prompt_version=1" in out.read_text()
