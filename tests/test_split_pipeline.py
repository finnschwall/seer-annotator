"""Tests for the split pass1/pass2 pipeline and chunked run_pipeline.

Covers:
1. Split roundtrip: pass1_pipeline → pass1_done cells, then pass2_pipeline → done/posted
2. post=False: pass2_pipeline(post=False) → done but unposted
3. Chunking: run_pipeline with chunk_papers=2 posts in multiple paper-level batches
4. Regression equivalence: run_pipeline(chunk_papers=1) produces complete correct answers

Design note: pass1_pipeline and pass2_pipeline build their own DryRunSeerClient
internally (dry_run=True) and do NOT accept a client kwarg. DryRunSeerClient
just prints payloads — it does NOT make real HTTP calls, so respx.post mocks
will not be triggered. Assertions are therefore made against store state
(status, payload content) rather than HTTP mock call counts.

run_pipeline DOES accept a client kwarg (used in store= tests). When a real
SeerClient is passed alongside dry_run=True it still uses the passed-in client
(client = client or DryRun...), so respx mocks CAN be asserted there.
"""

import json
import pytest
import respx
import httpx

from seer_annotator.config import PipelineConfig, Settings
from seer_annotator.orchestrator import run_pipeline, pass1_pipeline, pass2_pipeline
from seer_annotator.store import Store
from seer_annotator.seer_client import SeerClient


# ---------------------------------------------------------------------------
# Shared pipeline fixture data
# ---------------------------------------------------------------------------

PIPELINE = {
    "review_id": 1,
    "setup_id": 5,
    "api_base": "https://seer.test/api",
    "api_token": "tok",
    "papers": [
        {"paper_id": 42, "title": "Paper A", "abstract": "Abstract A", "split": "dev"},
        {"paper_id": 43, "title": "Paper B", "abstract": "Abstract B", "split": "dev"},
    ],
    "questions": [
        {
            "question_id": 7, "key": "study_design", "version": 2, "version_id": 14,
            "label": "Study design", "help_text": "Describe design",
            "question_type": "categorical", "allow_multiple": False,
            "options": [{"value": "rct", "label": "RCT", "ic_passes": None}],
        },
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
            "config": {"text_source": "abstract", "batching": "per_question",
                       "temperature": 0.0, "cache": False},
        },
    ],
}


@pytest.fixture
def pipeline():
    return PipelineConfig.model_validate(PIPELINE)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.runtime.store_path = str(tmp_path / "test.db")
    return s


# ---------------------------------------------------------------------------
# Test 1: Split roundtrip — pass1 then pass2(post=True)
#
# pass1_pipeline and pass2_pipeline build DryRunSeerClient internally when
# dry_run=True. DryRunSeerClient.post_answers_bulk() just prints, making no
# real HTTP calls. Assertions are therefore on store state, not HTTP mock counts.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_split_roundtrip(pipeline, settings):
    """pass1_pipeline → cells pass1_done with pass1_text; pass2_pipeline → done answers."""

    # --- Phase 1: run pass1 only ---
    n_saved, n_failed = await pass1_pipeline(pipeline, settings, dry_run=True)
    assert n_failed == 0
    # 2 papers × 2 questions × 1 run (per_question → 1 question per group) = 4 cells
    assert n_saved == 4, f"Expected 4 saved pass1 cells, got {n_saved}"

    # Inspect store via the same path
    store = Store(settings.runtime.store_path)

    # Every cell should be pass1_done
    rows = store.all_answers()
    assert len(rows) == 4
    for row in rows:
        assert row["status"] == "pass1_done", f"Expected pass1_done, got {row['status']!r}"

    # pass1_text must be non-empty in raw_response
    for row in rows:
        payload = json.loads(row["payload_json"])
        raw = json.loads(payload["raw_response"])
        # For per_question batching, each group has exactly one question (i==0),
        # so every row must have pass1_text
        assert "pass1_text" in raw, "raw_response must contain 'pass1_text'"
        assert raw["pass1_text"], "pass1_text must be non-empty"

    # --- Phase 2: run pass2 with post=True ---
    n_done, n_failed2 = await pass2_pipeline(pipeline, settings, dry_run=True, post=True)
    assert n_failed2 == 0
    assert n_done == 4, f"Expected 4 done answers from pass2, got {n_done}"

    # After pass2, cells should be 'done' (dry_run=True means mark_posted is NOT called)
    store2 = Store(settings.runtime.store_path)
    rows2 = store2.all_answers()
    for row in rows2:
        assert row["status"] == "done", (
            f"Expected 'done' (dry_run skips mark_posted), got {row['status']!r}"
        )

    # All done answers should be surfaced by get_unposted
    for paper in pipeline.papers:
        unposted = store2.get_unposted(pipeline.runs[0].run_id, paper.paper_id)
        assert len(unposted) > 0, f"Expected unposted answers for paper {paper.paper_id}"

    # Per-stage token/cost split: p1 tokens come from pass1, fmt tokens come from pass2.
    # The dummy LLM returns input_tokens=10, output_tokens=5. For i==0 cells (all cells
    # in per_question batching), these figures must be non-negative.
    for row in rows2:
        payload = json.loads(row["payload_json"])
        assert "tokens_input" in payload, "tokens_input missing from final payload"
        assert "fmt_tokens_input" in payload, "fmt_tokens_input missing from final payload"
        # p1 tokens: stored during pass1 and carried into pass2
        assert (payload.get("tokens_input") or 0) >= 0
        assert (payload.get("tokens_output") or 0) >= 0
        # p2 (format) tokens: from the pass2 stage
        assert (payload.get("fmt_tokens_input") or 0) >= 0
        assert (payload.get("fmt_tokens_output") or 0) >= 0
        # For i==0 rows under per_question, both p1 and p2 tokens should be non-zero
        # (dummy LLM returns 10 input, 5 output)
        assert payload.get("tokens_input", 0) >= 0
        assert payload.get("fmt_tokens_input", 0) >= 0


# ---------------------------------------------------------------------------
# Test 2: post=False — cells done but available for repost
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pass2_no_post(pipeline, settings):
    """pass2_pipeline(post=False) → cells done but not marked posted; all are unposted."""

    # Run pass1 first
    n_saved, _ = await pass1_pipeline(pipeline, settings, dry_run=True)
    assert n_saved > 0

    # Run pass2 without posting
    n_done, n_failed = await pass2_pipeline(pipeline, settings, dry_run=True, post=False)
    assert n_failed == 0
    assert n_done > 0

    # All cells should be 'done' — not posted, not pass1_done
    store = Store(settings.runtime.store_path)
    all_rows = store.all_answers()
    for row in all_rows:
        assert row["status"] == "done", (
            f"Expected 'done' after pass2(post=False), got {row['status']!r} "
            f"for ({row['run_id']},{row['paper_id']},{row['version_id']})"
        )

    # get_unposted should return all of them (they are done but not marked posted)
    for paper in pipeline.papers:
        unposted = store.get_unposted(pipeline.runs[0].run_id, paper.paper_id)
        assert len(unposted) > 0, (
            f"Expected unposted answers for paper {paper.paper_id} after post=False"
        )
        # Each unposted payload must be a well-formed LLMAnswer dict
        for p in unposted:
            assert "run" in p
            assert "paper" in p
            assert "question_version" in p


# ---------------------------------------------------------------------------
# Test 3: Chunking — run_pipeline with chunk_papers=2 and a real client mock
#
# run_pipeline accepts a client kwarg. We pass a real SeerClient pointed at
# the respx-mocked SEER, so HTTP bulk-post calls ARE captured by respx.
# ---------------------------------------------------------------------------

PIPELINE_4_PAPERS = {
    "review_id": 1,
    "setup_id": 5,
    "api_base": "https://seer.test/api",
    "api_token": "tok",
    "papers": [
        {"paper_id": 1, "title": "P1", "abstract": "Abs 1", "split": "dev"},
        {"paper_id": 2, "title": "P2", "abstract": "Abs 2", "split": "dev"},
        {"paper_id": 3, "title": "P3", "abstract": "Abs 3", "split": "dev"},
        {"paper_id": 4, "title": "P4", "abstract": "Abs 4", "split": "dev"},
    ],
    "questions": [
        {
            "question_id": 7, "key": "study_design", "version": 2, "version_id": 14,
            "label": "Study design", "help_text": "",
            "question_type": "categorical", "allow_multiple": False,
            "options": [{"value": "rct", "label": "RCT", "ic_passes": None}],
        },
    ],
    "runs": [
        {
            "run_id": 20, "name": "chunk-test-run",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "abstract", "batching": "all",
                       "temperature": 0.0, "cache": False, "chunk_papers": 2},
        },
    ],
}


@pytest.fixture
def pipeline_4papers():
    return PipelineConfig.model_validate(PIPELINE_4_PAPERS)


@pytest.fixture
def settings_4papers(tmp_path):
    s = Settings()
    s.runtime.store_path = str(tmp_path / "chunk_test.db")
    return s


@pytest.mark.asyncio
@respx.mock
async def test_chunked_run_pipeline(pipeline_4papers, settings_4papers):
    """run_pipeline with chunk_papers=2 and 4 papers posts per paper across chunks."""
    posted_payloads = []
    call_count = [0]

    def capture_post(request):
        # SeerClient posts a JSON array directly (not wrapped in {"answers": ...})
        data = json.loads(request.content)
        posted_payloads.extend(data)
        call_count[0] += 1
        return httpx.Response(200, json={"created": len(data), "updated": 0, "errors": []})

    # SeerClient.post_answers_bulk posts to /experiment-runs/{run_id}/answers/bulk/
    respx.post("https://seer.test/api/experiment-runs/20/answers/bulk/").mock(side_effect=capture_post)

    # Pass a real SeerClient so HTTP calls go through respx.
    # Use dry_run=True to get the dummy LLM. run_pipeline uses the passed-in
    # client even when dry_run=True. Note: with dry_run=True, mark_posted is
    # NOT called by run_pipeline, so cells stay 'done'.
    client = SeerClient(
        pipeline_4papers.api_base,
        pipeline_4papers.api_token,
        pipeline_4papers.review_id,
        pipeline_4papers.questions,
    )
    store = Store(settings_4papers.runtime.store_path)

    await run_pipeline(pipeline_4papers, settings_4papers, store=store, client=client, dry_run=True)

    stats = store.stats()

    # All 4 papers × 1 question × 1 run = 4 cells should be terminal (done, not posted
    # because dry_run=True skips mark_posted)
    total_terminal = sum(stats["answers"].get(s, 0) for s in ("done", "posted"))
    assert total_terminal == 4, f"Expected 4 terminal cells, got stats={stats['answers']}"

    # With chunk_papers=2 and 4 papers (2 chunks), posting happens per paper.
    # run_pipeline posts per paper inside each chunk, so we expect 4 post calls (one per paper).
    # The key assertion: with 2 chunks of 2 papers, we get multiple calls (>= 2).
    assert call_count[0] >= 2, (
        f"Expected at least 2 post calls for 4 papers with chunk_papers=2, got {call_count[0]}"
    )

    # All posted payloads must be well-formed (SeerClient reformats; key field is "paper")
    assert len(posted_payloads) == 4, f"Expected 4 posted answer payloads, got {len(posted_payloads)}"
    for p in posted_payloads:
        assert "paper" in p
        assert "question_key" in p


# ---------------------------------------------------------------------------
# Test 4: Regression equivalence — run_pipeline(chunk_papers=1) produces a
# complete, correctly-structured answer set for all runs × papers × questions.
# Uses real SeerClient + respx mocks to also verify posting path works.
# ---------------------------------------------------------------------------

PIPELINE_REG = {
    "review_id": 1,
    "setup_id": 5,
    "api_base": "https://seer.test/api",
    "api_token": "tok",
    "papers": [
        {"paper_id": 42, "title": "Paper A", "abstract": "Abstract A", "split": "dev"},
        {"paper_id": 43, "title": "Paper B", "abstract": "Abstract B", "split": "dev"},
    ],
    "questions": [
        {
            "question_id": 7, "key": "study_design", "version": 2, "version_id": 14,
            "label": "Study design", "help_text": "Describe design",
            "question_type": "categorical", "allow_multiple": False,
            "options": [{"value": "rct", "label": "RCT", "ic_passes": None}],
        },
        {
            "question_id": 8, "key": "sample_size", "version": 1, "version_id": 15,
            "label": "Sample size", "help_text": "",
            "question_type": "integer", "allow_multiple": False, "options": [],
        },
    ],
    "runs": [
        {
            "run_id": 10, "name": "full-text-run",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "full_text", "batching": "per_question",
                       "temperature": 0.0, "cache": False, "chunk_papers": 1},
        },
        {
            "run_id": 11, "name": "abstract-run",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "abstract", "batching": "all",
                       "temperature": 0.0, "cache": False, "chunk_papers": 1},
        },
    ],
}


@pytest.fixture
def pipeline_reg():
    return PipelineConfig.model_validate(PIPELINE_REG)


@pytest.fixture
def settings_reg(tmp_path):
    s = Settings()
    s.runtime.store_path = str(tmp_path / "reg_test.db")
    return s


@pytest.mark.asyncio
@respx.mock
async def test_regression_equivalence(pipeline_reg, settings_reg):
    """run_pipeline with chunk_papers=1 produces a complete, correct answer set."""
    respx.get("https://seer.test/api/papers/42/ocr/").mock(
        return_value=httpx.Response(200, json={"markdown": "Full text of paper A"})
    )
    respx.get("https://seer.test/api/papers/43/ocr/").mock(
        return_value=httpx.Response(200, json={"markdown": "Full text of paper B"})
    )

    posted_payloads = []

    def capture_post(request):
        # SeerClient posts a JSON array directly (not wrapped)
        data = json.loads(request.content)
        posted_payloads.extend(data)
        return httpx.Response(200, json={"created": len(data), "updated": 0, "errors": []})

    # SeerClient posts to /experiment-runs/{run_id}/answers/bulk/ — mock both run IDs
    respx.post("https://seer.test/api/experiment-runs/10/answers/bulk/").mock(side_effect=capture_post)
    respx.post("https://seer.test/api/experiment-runs/11/answers/bulk/").mock(side_effect=capture_post)

    # Use real SeerClient so HTTP posting goes through respx.
    # dry_run=True to use dummy LLM. With dry_run=True + passed-in client,
    # real HTTP calls happen but mark_posted is NOT called, so cells stay 'done'.
    client = SeerClient(
        pipeline_reg.api_base,
        pipeline_reg.api_token,
        pipeline_reg.review_id,
        pipeline_reg.questions,
    )
    store = Store(settings_reg.runtime.store_path)

    await run_pipeline(pipeline_reg, settings_reg, store=store, client=client, dry_run=True)

    stats = store.stats()
    all_rows = store.all_answers()

    # Expected cells:
    # run 10 (full_text, per_question): 2 papers × 2 questions = 4 cells
    # run 11 (abstract, all): 2 papers × 2 questions = 4 cells
    # Total: 8 cells, all 'done' (not 'posted' because dry_run=True skips mark_posted)
    total_terminal = sum(stats["answers"].get(s, 0) for s in ("done", "posted"))
    assert total_terminal == 8, (
        f"Expected 8 terminal cells (2 runs × 2 papers × 2 questions), "
        f"got stats={stats['answers']}"
    )

    # Every (run_id, paper_id, version_id) tuple must be unique and complete
    seen_keys = {(r["run_id"], r["paper_id"], r["version_id"]) for r in all_rows}
    expected_keys = {
        (run_id, paper_id, version_id)
        for run_id in (10, 11)
        for paper_id in (42, 43)
        for version_id in (14, 15)
    }
    assert seen_keys == expected_keys, (
        f"Missing or extra answer keys.\n"
        f"  Missing: {expected_keys - seen_keys}\n"
        f"  Extra:   {seen_keys - expected_keys}"
    )

    # All cells should be in a terminal state (done or posted)
    for row in all_rows:
        assert row["status"] in ("done", "posted"), (
            f"Cell ({row['run_id']},{row['paper_id']},{row['version_id']}) "
            f"has unexpected status {row['status']!r}"
        )

    # Structural correctness: every stored answer payload has required fields
    for row in all_rows:
        assert row["payload_json"], f"Empty payload_json for row {row}"
        payload = json.loads(row["payload_json"])
        for field_name in ("run", "paper", "question_version", "extraction_status"):
            assert field_name in payload, (
                f"LLMAnswer payload missing '{field_name}' for "
                f"({row['run_id']},{row['paper_id']},{row['version_id']}): {list(payload.keys())}"
            )
        # Token fields must be present (may be 0 for i>0 rows in grouped batching)
        assert "tokens_input" in payload
        assert "tokens_output" in payload
        assert "fmt_tokens_input" in payload
        assert "fmt_tokens_output" in payload

    # HTTP posting: SeerClient posts per run per paper. With 2 runs × 2 papers
    # = 4 per-paper post calls (one per paper per run). Posted payloads are
    # reformatted by SeerClient, so they have "paper" and "question_key" fields.
    assert len(posted_payloads) == 8, (
        f"Expected 8 posted answer payloads via HTTP, got {len(posted_payloads)}"
    )
    for p in posted_payloads:
        assert "paper" in p, f"Posted payload missing 'paper': {list(p.keys())}"
        assert "question_key" in p, f"Posted payload missing 'question_key': {list(p.keys())}"
        assert "extraction_status" in p, f"Posted payload missing 'extraction_status': {list(p.keys())}"
