"""Arbitration integration tests: mocked SEER + dummy LLM, full round-trip."""

import json

import httpx
import pytest
import respx

from seer_annotator.arbitrate.prompt import build_dispute_messages
from seer_annotator.arbitrate_orchestrator import run_arbitration_pipeline
from seer_annotator.config import DisputePipelineConfig, Settings
from seer_annotator.seer_client import SeerClient
from seer_annotator.store import Store

DISPUTE_PIPELINE = {
    "review_id": 1,
    "review_name": "Test review",
    "dispute_set_id": 3,
    "dispute_set_name": "round-2",
    "api_base": "https://seer.test/api",
    "api_token": "tok",
    "rater_keys": ["user:3", "run:10"],
    "runs": [
        {
            "run_id": 21, "name": "gpt-4o-arbiter",
            "model_name": "gpt-4o", "model_provider": "openai",
            "config": {"text_source": "full_text", "batching": "all", "anonymize_raters": True},
        },
    ],
    "questions": [
        {
            "question_id": 7, "key": "study_design", "version": 2, "version_id": 14,
            "label": "Study design", "help_text": "", "question_type": "categorical",
            "allow_multiple": False, "is_ic": False,
            "options": [{"value": "rct", "label": "RCT"}, {"value": "cohort", "label": "Cohort"}],
        },
        {
            "question_id": 8, "key": "sample_size", "version": 1, "version_id": 15,
            "label": "Sample size", "help_text": "", "question_type": "text",
            "allow_multiple": False, "is_ic": False, "options": [],
        },
    ],
    "disputes": [
        {
            "dispute_item_id": 501, "paper_id": 42, "paper_title": "Effects of metformin",
            "abstract": "Background...", "question_key": "study_design", "version_id": 14,
            "candidates": [
                {"rater_key": "user:3", "value": "rct", "comment": "Randomised per methods.",
                 "cited_text": "patients were randomly assigned", "source_answer_id": 9001},
                {"rater_key": "run:10", "value": "cohort", "comment": "No randomisation mentioned.",
                 "cited_text": "", "source_answer_id": 8801},
            ],
        },
        {
            "dispute_item_id": 502, "paper_id": 42, "paper_title": "Effects of metformin",
            "abstract": "Background...", "question_key": "sample_size", "version_id": 15,
            "candidates": [
                {"rater_key": "user:3", "value": "120", "comment": "Stated in abstract.",
                 "cited_text": "120 patients", "source_answer_id": 9002},
                {"rater_key": "run:10", "value": "240", "comment": "Model reasoning.",
                 "cited_text": "240 participants", "source_answer_id": 8802},
            ],
        },
    ],
}


@pytest.fixture
def pipeline():
    return DisputePipelineConfig.model_validate(DISPUTE_PIPELINE)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.runtime.store_path = str(tmp_path / "test.db")
    return s


@pytest.mark.asyncio
@respx.mock
async def test_arbitration_roundtrip_posts_resolutions(pipeline, settings):
    respx.get("https://seer.test/api/papers/42/ocr/").mock(
        return_value=httpx.Response(200, json={"markdown": "Full text of the paper"})
    )

    posted = []

    def capture(request):
        data = json.loads(request.content)
        posted.extend(data)
        return httpx.Response(200, json={"created": len(data), "updated": 0, "errors": []})

    respx.post("https://seer.test/api/arbiter-runs/21/resolutions/bulk/").mock(side_effect=capture)

    store = Store(settings.runtime.store_path)
    client = SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    # dry_run=True selects the dummy LLM (no real API key needed) while the
    # explicitly-passed real `client` still posts to the mocked endpoint —
    # matching the pattern used by tests/test_integration.py.
    await run_arbitration_pipeline(pipeline, settings, store=store, client=client, dry_run=True)

    stats = store.resolution_stats()
    assert stats.get("done", 0) + stats.get("posted", 0) == 2
    assert {p["dispute_item"] for p in posted} == {501, 502}
    for item in posted:
        assert "cited_text_verified" not in item  # Resolution wire format has no such field


@pytest.mark.asyncio
async def test_candidates_only_never_fetches_ocr(pipeline, settings):
    """text_source='candidates_only' must not touch the OCR endpoint at all."""
    for run in pipeline.runs:
        run.config.text_source = "candidates_only"

    store = Store(settings.runtime.store_path)

    class ExplodingClient(SeerClient):
        async def fetch_ocr_markdown(self, paper_id: int):
            raise AssertionError("OCR should not be fetched in candidates_only mode")

    client = ExplodingClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    await run_arbitration_pipeline(pipeline, settings, store=store, client=client, dry_run=True)

    assert store.resolution_stats().get("done") == 2


@pytest.mark.asyncio
async def test_idempotent_rerun_skips_posted_resolutions(pipeline, settings):
    from seer_annotator.llm import dummy_complete

    store = Store(settings.runtime.store_path)
    call_count = [0]

    async def counting_complete(*args, **kwargs):
        call_count[0] += 1
        return await dummy_complete(*args, **kwargs)

    for d in pipeline.disputes:
        store.save_resolution(21, d.dispute_item_id, d.paper_id, d.version_id, {"resolution_status": "ok"})
        store.mark_resolutions_posted(21, [d.dispute_item_id])

    import seer_annotator.llm as llm_mod
    original = llm_mod.complete
    llm_mod.complete = counting_complete
    try:
        for run in pipeline.runs:
            run.config.text_source = "candidates_only"
        await run_arbitration_pipeline(pipeline, settings, store=store, dry_run=False)
    finally:
        llm_mod.complete = original

    assert call_count[0] == 0, "No LLM calls expected when all disputes are already posted"


def test_anonymize_raters_hides_rater_key():
    from seer_annotator.config import Candidate, Question

    q = Question(question_id=1, key="k", version=1, version_id=1, label="L", question_type="text")
    candidates = [Candidate(rater_key="user:3", value="a", comment="c1"), Candidate(rater_key="run:10", value="b", comment="c2")]

    anon_messages = build_dispute_messages("paper text", [q], {1: candidates}, anonymize_raters=True, text_source="full_text")
    anon_text = " ".join(m["content"] for m in anon_messages if isinstance(m["content"], str))
    assert "user:3" not in anon_text and "run:10" not in anon_text
    assert "Candidate A" in anon_text and "Candidate B" in anon_text

    attributed_messages = build_dispute_messages("paper text", [q], {1: candidates}, anonymize_raters=False, text_source="full_text")
    attributed_text = " ".join(m["content"] for m in attributed_messages if isinstance(m["content"], str))
    assert "user:3" in attributed_text and "run:10" in attributed_text


def test_continuation_prompt_keeps_explicit_stopped_candidate_slot():
    from seer_annotator.config import Candidate, Question

    q = Question(
        question_id=2, key="follow_up", version=1, version_id=2,
        label="Follow up", question_type="boolean",
    )
    candidates = [
        Candidate(
            rater_key="run:1", answer_status="excluded",
            stopped_at_question_key="eligibility", detail="Paper excluded by this rater.",
        ),
        Candidate(rater_key="run:2", answer_status="answered", value=True),
    ]
    messages = build_dispute_messages(
        "paper", [q], {2: candidates}, item_types_by_version_id={2: "continuation"},
    )
    rendered = " ".join(m["content"] for m in messages if isinstance(m["content"], str))
    assert "CONTINUATION QUESTION" in rendered
    assert "No answer: Paper excluded by this rater. (at eligibility)" in rendered
    assert "Proposed value: True" in rendered


def test_adjudication_scope_accepts_omission_only_after_exclusion():
    from seer_annotator.arbitrate_orchestrator import _apply_adjudication_scope
    from seer_annotator.config import Question

    gate = Question(
        question_id=1, key="gate", version=1, version_id=1, label="Gate",
        question_type="boolean", is_ic=True, ic_include_when_true=True,
    )
    later = Question(
        question_id=2, key="later", version=1, version_id=2, label="Later",
        question_type="text",
    )
    stopped = _apply_adjudication_scope(
        [gate, later],
        [{"key": "gate", "value": False, "status": "ok"},
         {"key": "later", "value": None, "status": "absent"}],
    )
    assert stopped[1]["resolution_status"] == "not_applicable"

    continued = _apply_adjudication_scope(
        [gate, later],
        [{"key": "gate", "value": True, "status": "ok"},
         {"key": "later", "value": None, "status": "absent"}],
    )
    assert continued[1]["resolution_status"] == "error"


@pytest.mark.asyncio
@respx.mock
async def test_run_arbitration_pipeline_sends_start_and_terminal_heartbeats(pipeline, settings):
    """Mirrors test_progress.py's test_run_pipeline_sends_start_and_terminal_heartbeats,
    but for arbitrate's --progress-url wiring (one cell == one dispute item)."""
    respx.get("https://seer.test/api/papers/42/ocr/").mock(
        return_value=httpx.Response(200, json={"markdown": "Full text of the paper"})
    )
    respx.post("https://seer.test/api/arbiter-runs/21/resolutions/bulk/").mock(
        return_value=httpx.Response(200, json={"created": 2, "updated": 0, "errors": []})
    )

    heartbeats = []

    def capture(request):
        heartbeats.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    respx.post("https://seer.test/api/arbitration-jobs/7/progress/").mock(side_effect=capture)

    store = Store(settings.runtime.store_path)
    client = SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    await run_arbitration_pipeline(
        pipeline, settings, store=store, client=client, dry_run=True,
        progress_url="https://seer.test/api/arbitration-jobs/7/progress/",
    )

    assert len(heartbeats) >= 2
    assert heartbeats[0]["status"] == "running"
    assert heartbeats[0]["cells_done"] == 0
    assert heartbeats[0]["cells_total"] == 2  # one cell per dispute item, not paper x question
    assert heartbeats[-1]["status"] == "succeeded"
    assert heartbeats[-1]["cells_done"] == 2
    assert heartbeats[-1]["cells_error"] == 0


@pytest.mark.asyncio
async def test_run_arbitration_pipeline_without_progress_url_makes_no_heartbeat_calls(pipeline, settings):
    """progress_url=None (the default) must not attempt any HTTP calls for heartbeats."""
    store = Store(settings.runtime.store_path)

    class NoHeartbeatClient(SeerClient):
        async def post_resolutions_bulk(self, resolutions):
            return None

    client = NoHeartbeatClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)

    for run in pipeline.runs:
        run.config.text_source = "candidates_only"

    # No respx.mock active at all — any stray HTTP call (including a heartbeat)
    # would raise since httpx has no transport configured for this host in tests.
    await run_arbitration_pipeline(pipeline, settings, store=store, client=client, dry_run=True)


def test_candidates_only_omits_paper_text():
    from seer_annotator.config import Candidate, Question

    q = Question(question_id=1, key="k", version=1, version_id=1, label="L", question_type="text")
    candidates = [Candidate(rater_key="user:3", value="a", comment="c1")]

    messages = build_dispute_messages(
        "THIS SHOULD NOT APPEAR", [q], {1: candidates}, text_source="candidates_only",
    )
    full_text = " ".join(m["content"] for m in messages if isinstance(m["content"], str))
    assert "THIS SHOULD NOT APPEAR" not in full_text


@pytest.mark.asyncio
async def test_each_run_in_one_dispute_set_gets_its_own_rendering(settings):
    """Two arbiter runs, one job, one client, different text per run.

    A dispute set is adjudicated by several `ArbiterRun`s driven through a single
    client, and each has its own `exclude_sections` — what is withheld from the
    model is a per-run decision. Before the client protocol carried the run, the
    first run to ask for a paper fixed what every later run in the job was sent.
    """
    import seer_annotator.llm as llm_mod

    payload = json.loads(json.dumps(DISPUTE_PIPELINE))
    second = json.loads(json.dumps(payload["runs"][0]))
    second.update({"run_id": 22, "name": "gpt-4o-arbiter-with-refs"})
    third = json.loads(json.dumps(payload["runs"][0]))
    third.update({"run_id": 23, "name": "gpt-4o-arbiter-twin"})
    payload["runs"] = [payload["runs"][0], second, third]
    pipeline = DisputePipelineConfig.model_validate(payload)

    # Run 23 renders exactly like run 21 — it must share the cache entry, not
    # pay for a second fetch.
    variants = {21: "no-refs", 22: "whole", 23: "no-refs"}
    texts = {
        "no-refs": "BODY ONLY, references withheld",
        "whole": "BODY AND THE WHOLE REFERENCE LIST",
    }

    class PerRunClient(SeerClient):
        def __init__(self):
            super().__init__(pipeline.api_base, pipeline.api_token,
                             pipeline.review_id, pipeline.questions)
            self.fetches = []

        def ocr_variant(self, run_id: int) -> str:
            return variants[run_id]

        async def fetch_ocr_markdown(self, paper_id: int, run_id: int):
            self.fetches.append((paper_id, run_id))
            return texts[variants[run_id]]

        async def post_resolutions_bulk(self, resolutions):
            return None

    client = PerRunClient()

    prompts: list[str] = []

    async def recording_complete(model, provider, messages, **kw):
        prompts.append(
            " ".join(m["content"] for m in messages if isinstance(m.get("content"), str))
        )
        return await llm_mod.dummy_complete(model, provider, messages, **kw)

    original = llm_mod.complete
    llm_mod.complete = recording_complete
    try:
        store = Store(settings.runtime.store_path)
        await run_arbitration_pipeline(pipeline, settings, store=store, client=client)
    finally:
        llm_mod.complete = original

    # One fetch per *rendering*, not per run and not once for the whole job.
    assert client.fetches == [(42, 21), (42, 22)]

    # Every run's own text, and only its own, reached the model. Pass 2 is sent
    # Pass 1's reply rather than the paper, so only the prompts carrying paper
    # text at all are counted: one per run, three runs.
    carrying = [p for p in prompts if texts["no-refs"] in p or texts["whole"] in p]
    with_refs = [p for p in carrying if texts["whole"] in p]
    without_refs = [p for p in carrying if texts["whole"] not in p]
    assert len(carrying) == 3, "expected one Pass-1 prompt with paper text per run"
    assert len(with_refs) == 1, "only run 22 asked for the reference list"
    assert len(without_refs) == 2, "runs 21 and 23 asked for the paper without it"
    for prompt in without_refs:
        assert texts["no-refs"] in prompt
        assert "REFERENCE LIST" not in prompt
