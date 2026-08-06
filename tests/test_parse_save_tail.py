"""Tests for orchestrator._parse_save_post_tail's per-key parse tolerance (A5)
and deterministic skip/error relabeling (A4), against synthetic Pass-2 JSON —
no network, no live LLM. Uses a real (tmp-file) Store since that's the actual
idempotency boundary, but everything else is pure/local.
"""

import json

import pytest

from seer_annotator.config import ExperimentRun, Question, RunConfig
from seer_annotator.orchestrator import _parse_save_post_tail
from seer_annotator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def _gate_question(vid=1):
    return Question(
        question_id=vid, key="gate", version=1, version_id=vid, label="Gate",
        question_type="boolean", is_ic=True, ic_include_when_true=True,
    )


def _downstream_question(vid=2, key="downstream"):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type="text",
    )


def _run(early_exit: bool) -> ExperimentRun:
    return ExperimentRun(
        run_id=1, name="r", model_name="m", model_provider="anthropic",
        config=RunConfig(early_exit_on_ic_exclusion=early_exit, batching="all"),
    )


def _wire(results):
    return json.dumps({"results": results})


def _answers_by_key(store, run_id, paper_id, question_map):
    rows = store.all_answers(run_id=run_id, paper_id=paper_id)
    out = {}
    for r in rows:
        payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
        key = question_map.get(r["version_id"])
        out[key] = payload
    return out


def test_downstream_marked_skipped_when_early_exit_enabled(store):
    gate = _gate_question(1)
    downstream = _downstream_question(2)
    group = [gate, downstream]
    run = _run(early_exit=True)
    cfg = run.config

    p2_text = _wire([
        {"key": "gate", "value": False, "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
        {"key": "downstream", "value": "the model answered anyway", "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
    ])
    pending_cells = {"cid1": (type("P", (), {"paper_id": 100})(), group, 0)}

    err = _parse_save_post_tail(
        p1_texts={"cid1": "p1 text"}, p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending_cells, source_texts={100: "source"},
        run=run, cfg=cfg, store=store, fail_fast=False,
    )
    assert err is None

    qmap = {1: "gate", 2: "downstream"}
    answers = _answers_by_key(store, 1, 100, qmap)
    assert answers["gate"]["extraction_status"] == "ok"
    assert answers["gate"]["value_boolean"] is False
    assert answers["downstream"]["extraction_status"] == "skipped"
    assert answers["downstream"]["value_text"] is None
    assert "gate" in answers["downstream"]["extraction_detail"]


def test_downstream_not_skipped_when_flag_disabled(store):
    """Behavior-preservation guarantee: with early_exit_on_ic_exclusion off,
    an IC exclusion in the data must NOT produce a skipped row."""
    gate = _gate_question(1)
    downstream = _downstream_question(2)
    group = [gate, downstream]
    run = _run(early_exit=False)
    cfg = run.config

    p2_text = _wire([
        {"key": "gate", "value": False, "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
        {"key": "downstream", "value": "a real answer", "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
    ])
    pending_cells = {"cid1": (type("P", (), {"paper_id": 101})(), group, 0)}

    _parse_save_post_tail(
        p1_texts={"cid1": "p1 text"}, p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending_cells, source_texts={101: "source"},
        run=run, cfg=cfg, store=store, fail_fast=False,
    )

    qmap = {1: "gate", 2: "downstream"}
    answers = _answers_by_key(store, 1, 101, qmap)
    assert answers["downstream"]["extraction_status"] == "ok"
    assert answers["downstream"]["value_text"] == "a real answer"


def test_one_missing_key_does_not_error_the_whole_group(store):
    """A5: a single P2-omitted key must not discard the OTHER key's good
    answer (the pre-existing whole-group ExtractionError bug)."""
    q1 = _downstream_question(1, "q1")
    q2 = _downstream_question(2, "q2")
    group = [q1, q2]
    run = _run(early_exit=False)
    cfg = run.config

    # q2 entirely absent from Pass-2 output.
    p2_text = _wire([
        {"key": "q1", "value": "good answer", "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
    ])
    pending_cells = {"cid1": (type("P", (), {"paper_id": 102})(), group, 0)}

    _parse_save_post_tail(
        p1_texts={"cid1": "p1 text"}, p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending_cells, source_texts={102: "source"},
        run=run, cfg=cfg, store=store, fail_fast=False,
    )

    qmap = {1: "q1", 2: "q2"}
    answers = _answers_by_key(store, 1, 102, qmap)
    assert answers["q1"]["extraction_status"] == "ok"
    assert answers["q1"]["value_text"] == "good answer"
    assert answers["q2"]["extraction_status"] == "error"


def test_unmappable_status_becomes_invalid(store):
    q1 = _downstream_question(1, "q1")
    group = [q1]
    run = _run(early_exit=False)
    cfg = run.config

    p2_text = _wire([
        {"key": "q1", "value": None, "cited_text": "", "comment": "", "confidence": None, "status": "unmappable"},
    ])
    pending_cells = {"cid1": (type("P", (), {"paper_id": 103})(), group, 0)}

    _parse_save_post_tail(
        p1_texts={"cid1": "p1 text"}, p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending_cells, source_texts={103: "source"},
        run=run, cfg=cfg, store=store, fail_fast=False,
    )

    qmap = {1: "q1"}
    answers = _answers_by_key(store, 1, 103, qmap)
    assert answers["q1"]["extraction_status"] == "invalid"


def test_excluded_papers_out_param_populated(store):
    gate = _gate_question(1)
    downstream = _downstream_question(2)
    group = [gate, downstream]
    run = _run(early_exit=True)
    cfg = run.config

    p2_text = _wire([
        {"key": "gate", "value": False, "cited_text": "", "comment": "", "confidence": None, "status": "ok"},
        {"key": "downstream", "value": None, "cited_text": "", "comment": "", "confidence": None, "status": "absent"},
    ])
    pending_cells = {"cid1": (type("P", (), {"paper_id": 104})(), group, 0)}
    excluded: dict = {}

    _parse_save_post_tail(
        p1_texts={"cid1": "p1 text"}, p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending_cells, source_texts={104: "source"},
        run=run, cfg=cfg, store=store, fail_fast=False,
        excluded_papers=excluded,
    )

    assert 104 in excluded
    assert "gate" in excluded[104]
