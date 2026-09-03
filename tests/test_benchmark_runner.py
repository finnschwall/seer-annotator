import json
import sqlite3

import pytest

from seer_annotator.benchmarking import BenchmarkCase, BenchmarkStore, DatasetSpec, ModelConfig
from seer_annotator.benchmarking.models import load_model_configs
from seer_annotator.benchmarking.runner import BenchmarkRunner
from seer_annotator.annotate.parse import parse_structured_output_diagnostic


QUESTION = {
    "question_id": 1, "key": "q1", "version": 1, "version_id": 1,
    "label": "Question", "question_type": "boolean",
}


def _store(path, n=2):
    store = BenchmarkStore(path)
    store.create_dataset(DatasetSpec("d", "frozen", {"runs": [1]}))
    for i in range(n):
        store.add_case(BenchmarkCase(
            "d", f"case-{i}", 1, i, None, f"group-{i}", "Paper", "source",
            "abstract", "Pass 1", [QUESTION],
        ))
    return store


def _response(value=True):
    return json.dumps({"results": [{
        "key": "q1", "value": value, "status": "ok", "cited_text": "",
        "comment": "", "confidence": 10,
    }]})


@pytest.mark.asyncio
async def test_two_configs_accumulate_and_resume(tmp_path):
    store = _store(tmp_path / "bench.sqlite")
    calls = []

    async def complete(model, provider, messages, **kwargs):
        calls.append((model, provider, kwargs["response_format"], kwargs))
        return {"text": _response(model == "a"), "usage": {"input_tokens": 2}, "cost": 0.1}

    runner = BenchmarkRunner(store, completion_fn=complete, concurrency=2)
    a = ModelConfig("a", "provider", "model-a")
    b = ModelConfig("b", "provider", "model-b")
    assert (await runner.run("d", a))["complete"] == 2
    assert (await runner.run("d", b))["complete"] == 2
    assert len(calls) == 4
    assert all(call[2] is not None for call in calls)
    assert all("require_status" not in str(call[2]) for call in calls)
    assert all(call[3]["timeout"] == 600.0 for call in calls)
    assert (await runner.run("d", a))["skipped"] == 2
    assert len(calls) == 4

    rows = store.get_executions("d")
    assert {row["model_name"] for row in rows} == {"a", "b"}
    assert all(json.loads(row["usage_json"])["input_tokens"] == 2 for row in rows)


@pytest.mark.asyncio
async def test_provider_error_retries_only_when_requested(tmp_path):
    store = _store(tmp_path / "bench.sqlite", n=1)
    attempts = 0

    async def complete(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider unavailable")

    runner = BenchmarkRunner(store, completion_fn=complete)
    config = ModelConfig("err", "provider", "model")
    assert (await runner.run("d", config)) == {"complete": 0, "error": 1, "skipped": 0}
    assert (await runner.run("d", config)) == {"complete": 0, "error": 0, "skipped": 1}
    assert (await runner.run("d", config, retry_errors=True))["error"] == 1
    assert attempts == 2


def test_diagnostics_distinguish_absent_and_omitted():
    text = json.dumps({"results": [{
        "key": "q1", "value": None, "status": "absent", "cited_text": "",
        "comment": "", "confidence": None,
    }, {"key": "q1", "value": True}]})
    diagnostic = parse_structured_output_diagnostic(text, ["q1", "q2"])
    assert diagnostic["native_json"] is True
    assert diagnostic["duplicate_keys"] == ["q1"]
    assert diagnostic["missing_keys"] == ["q2"]
    assert diagnostic["answers"][0]["present"] is True
    assert diagnostic["answers"][1]["present"] is False


def test_diagnostics_record_fallback_and_unexpected():
    text = '{"key":"q1","value":true,"status":"ok"}\n' \
        '{"key":"other","value":false,}'
    diagnostic = parse_structured_output_diagnostic(text, ["q1"])
    assert diagnostic["native_json"] is False
    assert diagnostic["fallback_used"] is True
    assert diagnostic["repair_used"] is True
    assert diagnostic["unexpected_keys"] == ["other"]


def test_model_toml_loading(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text("[models.reference]\nprovider='p'\nmodel='m'\ntemperature=0\nfoo=1\n")
    config = load_model_configs(path)["reference"]
    assert config.params == {"foo": 1}
    assert config.config_hash


@pytest.mark.asyncio
async def test_config_redaction_preserves_token_metrics(tmp_path):
    store = _store(tmp_path / "bench.sqlite", n=1)
    config = ModelConfig("m", "p", "model", params={"max_tokens": 123, "api_key": "do-not-store"})
    store.add_model_config(config)
    row = store.get_model_config("m")
    saved = json.loads(row["config_json"])
    assert saved["params"]["max_tokens"] == 123
    assert saved["params"]["api_key"] == "[redacted]"

    seen = {}
    async def complete(*args, **kwargs):
        seen.update(kwargs)
        return {"text": _response()}
    await BenchmarkRunner(store, completion_fn=complete).run(
        "d", config,
    )
    assert seen["max_tokens"] == 123
    assert "api_key" not in seen


def test_stale_running_execution_is_recoverable(tmp_path):
    store = _store(tmp_path / "bench.sqlite", n=1)
    config = ModelConfig("m", "p", "model")
    config_id = store.add_model_config(config)
    store.ensure_executions("d", config_id)
    execution = store.get_executions("d", config_id)[0]
    assert store.claim_execution(execution["id"], run_token="old")
    with store.transaction() as con:
        con.execute("UPDATE executions SET claimed_at='2000-01-01T00:00:00+00:00' WHERE id=?", (execution["id"],))
    assert store.recover_stale_running(1) == 1
    assert store.get_execution(execution["case_id"], config_id)["status"] == "pending"
