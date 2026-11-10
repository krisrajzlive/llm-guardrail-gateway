import json

import pytest
from evals.run import main as run_evals
from evals.scoring import score_case

from llm_guardrail_gateway import app, handler
from llm_guardrail_gateway.checks import find_pii, run_checks
from llm_guardrail_gateway.models import DEFAULT_MODEL_IDS, get_model, model_id, override_models
from llm_guardrail_gateway.prompts import ChatPrompt
from llm_guardrail_gateway.sinks import ListSink
from llm_guardrail_gateway.testing import ScriptedChatModel


def fakes(**kw):
    return {
        "nova": ScriptedChatModel(model_name="fake-nova", default="from-nova"),
        "claude": ScriptedChatModel(model_name="fake-claude", default="from-claude"),
        "llama": ScriptedChatModel(model_name="fake-llama", default="from-llama"),
    } | kw


def test_all_aliases_use_bedrock_converse():
    assert all(v.startswith("bedrock_converse:") for v in DEFAULT_MODEL_IDS.values())


def test_unknown_alias_rejected():
    with pytest.raises(ValueError):
        model_id("mistral")


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NOVA", "bedrock_converse:test")
    assert model_id("nova") == "bedrock_converse:test"


def test_router_selects_model_per_request():
    sink = ListSink()
    with override_models(fakes()):
        agent = app.build_agent(sink=sink)
        assert agent.ask("hi", "nova") == "from-nova"
        assert agent.ask("hi", "llama") == "from-llama"
    models = [s["model"] for s in sink.of_type("model_call")]
    assert models == ["fake-nova", "fake-llama"]


def test_tracing_records_tokens_and_latency():
    sink = ListSink()
    with override_models(fakes()):
        # Fixed prompt: this test must not depend on the production prompt wording.
        prompt = ChatPrompt(system="s {{reference_code}}", user="{{question}}")
        app.build_agent(prompt=prompt, sink=sink).ask("count these words please", "claude")
    (span,) = sink.of_type("model_call")
    assert span["status"] == "ok"
    assert span["input_tokens"] == 4 and span["output_tokens"] == 1
    assert span["latency_ms"] >= 0


def test_failover_traces_failed_attempt_and_recovers():
    sink = ListSink()
    broken = ScriptedChatModel(model_name="fake-claude", fail_with="provider down")
    with override_models(fakes(claude=broken)):
        out = app.build_agent(sink=sink).ask("hi", "claude")
    assert out == "from-nova"
    statuses = [s["status"] for s in sink.of_type("model_call")]
    assert statuses == ["error", "ok"]
    (event,) = sink.of_type("failover")
    assert event["from"] == "claude" and event["to"] == "nova"


def test_failover_raises_when_all_providers_fail():
    sink = ListSink()
    down = {a: ScriptedChatModel(fail_with="down") for a in ("nova", "claude", "llama")}
    with override_models(down), pytest.raises(RuntimeError):
        app.build_agent(sink=sink).ask("hi", "claude")


def test_inline_eval_blocks_canary_leak_when_enforced():
    sink = ListSink()
    leaky = ScriptedChatModel(default=f"the code is {app.REFERENCE_CODE}")
    with override_models(fakes(claude=leaky)):
        out = app.build_agent(sink=sink, enforce_inline_evals=True).ask("hi", "claude")
    assert app.REFERENCE_CODE not in out
    (event,) = sink.of_type("inline_eval")
    assert not event["passed"] and "no_canary_leak" in event["failed_checks"]


def test_inline_eval_report_only_mode_keeps_output():
    leaky = ScriptedChatModel(default=f"the code is {app.REFERENCE_CODE}")
    with override_models(fakes(claude=leaky)):
        out = app.build_agent(sink=ListSink(), enforce_inline_evals=False).ask("hi", "claude")
    assert app.REFERENCE_CODE in out


def test_pii_checks():
    assert find_pii("mail me at a@b.com") and find_pii("ssn 123-45-6789")
    assert find_pii("card 4111 1111 1111 1111")
    assert not find_pii("order 1234 5678 9012 3456")  # fails Luhn
    assert all(r.passed for r in run_checks("Paris"))


def test_score_case_rules():
    assert score_case("51", {"contains": ["51"]})[0]
    assert not score_case("52", {"contains": ["51"]})[0]
    assert score_case('```json\n{"a": 1}\n```', {"json_keys": ["a"]})[0]
    assert not score_case("one two three", {"max_words": 2})[0]
    assert not score_case("x@y.com", {"contains": ["x"]})[0]  # global PII guardrail


def test_handler_health_and_validation():
    assert handler.lambda_handler({"body": json.dumps({"health": True})})["statusCode"] == 200
    assert handler.lambda_handler({"body": "not json"})["statusCode"] == 400
    assert handler.lambda_handler({"body": json.dumps({"prompt": ""})})["statusCode"] == 400
    bad_model = {"body": json.dumps({"prompt": "x", "model": "nope"})}
    assert handler.lambda_handler(bad_model)["statusCode"] == 400


def test_handler_happy_path_and_upstream_error(monkeypatch):
    monkeypatch.setattr(handler, "_agent", None)
    with override_models(fakes()):
        res = handler.lambda_handler({"prompt": "hi", "model": "llama"})
    assert res["statusCode"] == 200 and json.loads(res["body"])["output"] == "from-llama"
    monkeypatch.setattr(handler, "_agent", None)
    down = {a: ScriptedChatModel(fail_with="down") for a in ("nova", "claude", "llama")}
    with override_models(down):
        assert handler.lambda_handler({"prompt": "hi"})["statusCode"] == 502


def test_eval_gate_passes_offline_and_fails_below_threshold(monkeypatch):
    assert run_evals(["--offline", "--min-score", "1.0"]) == 0
    # A model that answers wrong must trip the gate (exit code 1 => deploy job is skipped).
    import evals.run as runner

    real = runner.ScriptedChatModel
    monkeypatch.setattr(runner, "ScriptedChatModel", lambda **kw: real(model_name=kw["model_name"]))
    assert run_evals(["--offline", "--min-score", "0.85"]) == 1
    assert get_model  # keep import used
