"""Regression tests for the run-killing crash on a badly repaired Pass-2 item.

Prod failure (paper 2199): the model wrote a parenthetical outside a JSON string
inside a cited_text array. json_repair recovered all four items but resynchronised
badly — everything from the parenthetical on was swallowed into a nested list
inside cited_text, taking that item's comment/confidence/status with it.
verify_citation() then called .strip() on the nested list and raised
AttributeError, which escaped _parse_save_post_tail and ended the whole job:
146 of 246 papers were never attempted.

Three things are asserted here, matching the three separate weaknesses:
  1. the malformed item is flagged, not saved as a normal answer,
  2. verify_citation tolerates a non-string element instead of raising,
  3. one bad item costs its own cell only — the run keeps going.
"""

import json

import pytest

from seer_annotator.annotate.parse import (
    cited_text_violation,
    parse_structured_output,
)
from seer_annotator.annotate.verify import verify_citation, verify_citations
from seer_annotator.config import ExperimentRun, Question, RunConfig
from seer_annotator.orchestrator import _parse_save_post_tail
from seer_annotator.store import Store


# The shape json_repair produces from the prod text: three clean items, then one
# whose parenthetical broke the array and pulled the later fields in with it.
_REPAIRED_BAD_ITEM = {
    "key": "xai_applied_to_language_model",
    "value": "Yes",
    "cited_text": [
        "we theoretically characterize the representation capacity of Mamba variants",
        [
            "targeting the Mamba language model itself)",
            "even a single-layer Mamba efficiently learns in context",
            'comment": "The XAI-type analysis is applied to Mamba',
            {"confidence": 19, "status": "ok"},
        ],
    ],
}

_SOURCE = (
    "We theoretically characterize the representation capacity of Mamba variants. "
    "We further show that even a single-layer Mamba efficiently learns in context."
)


def _q(vid, key, qtype="text"):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type=qtype,
    )


def _p1_text(*keys):
    return "\n".join(
        f"--- ANSWER: {k} ---\nAnswer: something\nConfidence: 10\n" for k in keys
    )


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


# --- 1. the parser flags it -------------------------------------------------

def test_nested_list_in_cited_text_is_a_violation():
    detail = cited_text_violation(_REPAIRED_BAD_ITEM["cited_text"])
    assert detail is not None
    assert "non-string" in detail


def test_valid_shapes_are_not_violations():
    assert cited_text_violation(None) is None
    assert cited_text_violation("a quote") is None
    assert cited_text_violation(["a", "b"]) is None


def test_parse_flags_the_bad_item_and_keeps_the_good_ones():
    text = json.dumps({"results": [
        {"key": "a", "value": "Yes", "cited_text": "quote a", "comment": "", "confidence": 18, "status": "ok"},
        {"key": "b", "value": "No", "cited_text": ["quote b"], "comment": "", "confidence": 17, "status": "ok"},
        _REPAIRED_BAD_ITEM,
    ]})
    results = {r["key"]: r for r in parse_structured_output(text, ["a", "b", "xai_applied_to_language_model"], annotate_mode=True)}

    assert "cited_text_error" not in results["a"]
    assert "cited_text_error" not in results["b"]
    bad = results["xai_applied_to_language_model"]
    assert "cited_text_error" in bad
    # The nested element is dropped from the value handed downstream, so nothing
    # further along can trip over it even if it ignores the flag.
    assert bad["cited_text"] == [
        "we theoretically characterize the representation capacity of Mamba variants"
    ]


# --- 2. verification tolerates the shape ------------------------------------

def test_verify_citation_does_not_raise_on_a_nested_list():
    result = verify_citation(_REPAIRED_BAD_ITEM["cited_text"], _SOURCE)
    assert set(result) == {"ok", "note"}


def test_verify_citation_does_not_raise_on_a_dict():
    result = verify_citation({"quote": "x"}, _SOURCE)
    assert set(result) == {"ok", "note"}


def test_verify_citations_plural_unchanged_on_a_nested_list():
    results = verify_citations(_REPAIRED_BAD_ITEM["cited_text"], _SOURCE)
    assert len(results) == 2


# --- 3. one bad item costs one cell -----------------------------------------

def test_bad_item_becomes_an_error_answer_and_the_group_survives(store):
    group = [_q(1, "a"), _q(2, "b"), _q(3, "xai_applied_to_language_model")]
    run = ExperimentRun(
        run_id=1, name="r", model_name="m", model_provider="anthropic",
        config=RunConfig(batching="all"),
    )
    p2_text = json.dumps({"results": [
        {"key": "a", "value": "Yes", "cited_text": "", "comment": "", "confidence": 18, "status": "ok"},
        {"key": "b", "value": "No", "cited_text": "", "comment": "", "confidence": 17, "status": "ok"},
        _REPAIRED_BAD_ITEM,
    ]})
    pending = {"cid1": (type("P", (), {"paper_id": 2199})(), group, 0)}

    err = _parse_save_post_tail(
        p1_texts={"cid1": _p1_text("a", "b", "xai_applied_to_language_model")},
        p1_usage={}, p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending, source_texts={2199: _SOURCE},
        run=run, cfg=run.config, store=store, fail_fast=False,
    )
    assert err is None

    qmap = {1: "a", 2: "b", 3: "xai_applied_to_language_model"}
    answers = {}
    for row in store.all_answers(run_id=1, paper_id=2199):
        answers[qmap[row["version_id"]]] = json.loads(row["payload_json"])

    # All three cells got an answer — nothing vanished.
    assert set(answers) == {"a", "b", "xai_applied_to_language_model"}
    assert answers["a"]["extraction_status"] == "ok"
    assert answers["b"]["extraction_status"] == "ok"

    bad = answers["xai_applied_to_language_model"]
    assert bad["extraction_status"] == "error"
    assert bad["value_text"] is None
    assert "non-string" in bad["extraction_detail"]
    raw = bad["raw_response"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    assert [d["code"] for d in raw["diagnostics"]] == ["pass2_malformed_cited_text"]


def test_an_unexpected_exception_fails_one_cell_not_the_run(store, monkeypatch):
    """Belt and braces: whatever else can raise in the parse/verify tail, the
    run must not die. Before the fix, any exception here escaped asyncio.run."""
    from seer_annotator import orchestrator

    real_verify = orchestrator.verify_citation

    def exploding_verify(cited_text, source, **kw):
        if cited_text == "boom":
            raise RuntimeError("synthetic failure")
        return real_verify(cited_text, source, **kw)

    monkeypatch.setattr(orchestrator, "verify_citation", exploding_verify)

    group = [_q(1, "a"), _q(2, "b")]
    run = ExperimentRun(
        run_id=1, name="r", model_name="m", model_provider="anthropic",
        config=RunConfig(batching="all"),
    )
    p2_text = json.dumps({"results": [
        {"key": "a", "value": "Yes", "cited_text": "boom", "comment": "", "confidence": 18, "status": "ok"},
        {"key": "b", "value": "No", "cited_text": "", "comment": "", "confidence": 17, "status": "ok"},
    ]})
    pending = {"cid1": (type("P", (), {"paper_id": 2200})(), group, 0)}

    err = _parse_save_post_tail(
        p1_texts={"cid1": _p1_text("a", "b")}, p1_usage={},
        p2_texts={"cid1": p2_text}, p2_usage={},
        pending_cells=pending, source_texts={2200: _SOURCE},
        run=run, cfg=run.config, store=store, fail_fast=False,
    )
    assert err is None

    qmap = {1: "a", 2: "b"}
    answers = {}
    for row in store.all_answers(run_id=1, paper_id=2200):
        answers[qmap[row["version_id"]]] = json.loads(row["payload_json"])

    assert answers["a"]["extraction_status"] == "error"
    assert "synthetic failure" in answers["a"]["extraction_detail"]
    raw = answers["a"]["raw_response"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    assert [d["code"] for d in raw["diagnostics"]] == ["pass2_unexpected_error"]
    # The next question in the same group was still processed.
    assert answers["b"]["extraction_status"] == "ok"
