"""Tests for the heartbeat/progress client and its wiring into run_pipeline."""

import itertools
import json

import httpx
import pytest
import respx

from seer_annotator import orchestrator as orchestrator_module
from seer_annotator.config import PipelineConfig, Settings
from seer_annotator.orchestrator import run_pipeline
from seer_annotator.progress import ProgressReporter
from seer_annotator.seer_client import SeerClient
from seer_annotator.store import Store


@pytest.mark.asyncio
async def test_heartbeat_noop_without_url():
    reporter = ProgressReporter(None, "tok", 5)
    # Should return immediately without making any network call.
    await reporter.heartbeat(status="running", cells_total=1, cells_done=0, cells_error=0)


@pytest.mark.asyncio
@respx.mock
async def test_heartbeat_posts_expected_shape():
    """When no chunk/phase kwargs are given, those keys must be omitted from
    the POSTed JSON entirely — not sent as an explicit `null` — since the
    receiver treats key-absence, not null-ness, as "no update to this
    dimension" (see the `heartbeat` docstring)."""
    route = respx.post("https://seer.test/annotation-jobs/10/progress/").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    reporter = ProgressReporter("https://seer.test/annotation-jobs/10/progress/", "tok123", 10)

    await reporter.heartbeat(status="running", cells_total=10, cells_done=2, cells_error=1, message="chunk 1/5")

    assert route.call_count == 1
    request = route.calls[0].request
    assert request.headers["authorization"] == "Token tok123"
    body = json.loads(request.content)
    assert body == {
        "run_id": 10,
        "status": "running",
        "cells_total": 10,
        "cells_done": 2,
        "cells_error": 1,
        "message": "chunk 1/5",
    }
    for key in ("chunk_index", "chunk_total", "phase", "phase_done", "phase_total"):
        assert key not in body


@pytest.mark.asyncio
@respx.mock
async def test_heartbeat_posts_phase_fields_when_given():
    respx.post("https://seer.test/annotation-jobs/10/progress/").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    reporter = ProgressReporter("https://seer.test/annotation-jobs/10/progress/", "tok123", 10)

    await reporter.heartbeat(
        status="running", cells_total=10, cells_done=2, cells_error=1,
        chunk_index=1, chunk_total=5, phase="pass2", phase_done=3, phase_total=4,
    )

    request = respx.calls[0].request
    body = json.loads(request.content)
    assert body["chunk_index"] == 1
    assert body["chunk_total"] == 5
    assert body["phase"] == "pass2"
    assert body["phase_done"] == 3
    assert body["phase_total"] == 4


@pytest.mark.asyncio
@respx.mock
async def test_broken_progress_endpoint_does_not_raise():
    respx.post("https://seer.test/annotation-jobs/10/progress/").mock(return_value=httpx.Response(500))
    reporter = ProgressReporter("https://seer.test/annotation-jobs/10/progress/", "tok", 10)
    # Must not raise even though the endpoint 500s.
    await reporter.heartbeat(status="failed", cells_total=1, cells_done=0, cells_error=1)


PIPELINE = {
    "review_id": 1,
    "setup_id": 5,
    "api_base": "https://seer.test/api",
    "api_token": "tok",
    "papers": [
        {"paper_id": 42, "title": "Paper A", "abstract": "Abstract A", "split": "dev"},
    ],
    "questions": [
        {
            "question_id": 8, "key": "sample_size", "version": 1, "version_id": 15,
            "label": "Sample size", "help_text": "",
            "question_type": "integer", "allow_multiple": False, "options": [],
        },
    ],
    "runs": [
        {
            "run_id": 10, "name": "abstract-run",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "abstract", "batching": "all", "chunk_papers": 1},
        },
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_run_pipeline_sends_start_and_terminal_heartbeats(tmp_path):
    pipeline = PipelineConfig.model_validate(PIPELINE)
    settings = Settings()
    settings.runtime.store_path = str(tmp_path / "test.db")

    heartbeats = []

    def capture(request):
        heartbeats.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    respx.post("https://seer.test/api/annotation-jobs/10/progress/").mock(side_effect=capture)
    respx.post("https://seer.test/api/experiment-runs/10/answers/bulk/").mock(
        return_value=httpx.Response(200, json={"created": 1, "updated": 0, "errors": []})
    )

    store = Store(settings.runtime.store_path)
    client = SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    await run_pipeline(
        pipeline, settings, store=store, client=client, dry_run=True,
        progress_url="https://seer.test/api/annotation-jobs/10/progress/",
    )

    assert len(heartbeats) >= 2
    assert heartbeats[0]["status"] == "running"
    assert heartbeats[0]["cells_done"] == 0
    assert heartbeats[0].get("phase") is None
    assert "phase" not in heartbeats[0]
    assert heartbeats[-1]["status"] == "succeeded"
    assert heartbeats[-1]["cells_done"] == 1
    assert heartbeats[-1]["cells_error"] == 0
    assert heartbeats[-1].get("phase") is None
    assert "phase" not in heartbeats[-1]


PIPELINE_TWO_PAPERS = {
    **PIPELINE,
    "papers": [
        {"paper_id": 42, "title": "Paper A", "abstract": "Abstract A", "split": "dev"},
        {"paper_id": 43, "title": "Paper B", "abstract": "Abstract B", "split": "dev"},
    ],
    "runs": [
        {
            "run_id": 10, "name": "abstract-run",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "abstract", "batching": "all", "chunk_papers": 2},
        },
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_run_pipeline_sends_mid_chunk_phase_heartbeats(tmp_path, monkeypatch):
    """Per-completion heartbeats mid-chunk should carry pass1/pass2 phase progress
    instead of leaving the UI frozen until the whole chunk (both passes) finishes."""
    # Defeat the 1.5s throttle so every per-cell completion actually posts.
    counter = itertools.count(step=10.0)
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: next(counter))

    pipeline = PipelineConfig.model_validate(PIPELINE_TWO_PAPERS)
    settings = Settings()
    settings.runtime.store_path = str(tmp_path / "test.db")

    heartbeats = []

    def capture(request):
        heartbeats.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    respx.post("https://seer.test/api/annotation-jobs/10/progress/").mock(side_effect=capture)
    respx.post("https://seer.test/api/experiment-runs/10/answers/bulk/").mock(
        return_value=httpx.Response(200, json={"created": 2, "updated": 0, "errors": []})
    )

    store = Store(settings.runtime.store_path)
    client = SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    await run_pipeline(
        pipeline, settings, store=store, client=client, dry_run=True,
        progress_url="https://seer.test/api/annotation-jobs/10/progress/",
    )

    pass1_heartbeats = [h for h in heartbeats if h.get("phase") == "pass1"]
    pass2_heartbeats = [h for h in heartbeats if h.get("phase") == "pass2"]
    assert pass1_heartbeats, "expected at least one mid-chunk pass1 heartbeat"
    assert pass2_heartbeats, "expected at least one mid-chunk pass2 heartbeat"
    assert all(h["chunk_index"] == 0 and h["chunk_total"] == 1 for h in pass1_heartbeats + pass2_heartbeats)
    assert any(h["phase_done"] > 0 for h in pass1_heartbeats)
    assert any(h["phase_done"] > 0 for h in pass2_heartbeats)


@pytest.mark.asyncio
async def test_run_pipeline_without_progress_url_makes_no_heartbeat_calls(tmp_path):
    """progress_url=None (the default) must not attempt any HTTP calls for heartbeats."""
    pipeline = PipelineConfig.model_validate(PIPELINE)
    settings = Settings()
    settings.runtime.store_path = str(tmp_path / "test.db")

    store = Store(settings.runtime.store_path)

    class NoHeartbeatClient(SeerClient):
        async def post_answers_bulk(self, answers):
            return None

    client = NoHeartbeatClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    # No respx.mock active at all — any stray HTTP call (including a heartbeat)
    # would raise since httpx has no transport configured for this host in tests.
    await run_pipeline(pipeline, settings, store=store, client=client, dry_run=True)
