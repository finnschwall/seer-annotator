"""Tests for the heartbeat/progress client and its wiring into run_pipeline."""

import json

import httpx
import pytest
import respx

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
    assert heartbeats[-1]["status"] == "succeeded"
    assert heartbeats[-1]["cells_done"] == 1
    assert heartbeats[-1]["cells_error"] == 0


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
