"""A Pass-2 reply that never finished must not be read for answers.

Prod failure: the format model stopped writing JSON partway through a group and
looped on one word until the output ceiling. json_repair closed the dangling
string and the open braces, so the document parsed, and every item that came
before the loop was stored as a normal answer. The item the loop started inside
was stored too: its `value` and `cited_text` had already been written, its
`comment` was the truncated fragment, its `confidence` was gone, and its
`status` — never written either — defaulted to "ok" in `_extract_result`.

238 replies in one database ended this way and 43 answers came out of them
marked ok. They counted toward IC decisions and IRR like any other answer.

`cited_text_violation` already made this judgment, but only for wreckage that
landed inside `cited_text`; a loop starting one field later walked past it. The
gate here is one step earlier and does not depend on where the damage landed:
if the *reply* did not finish, no item in it is trustworthy, because which
items survived is decided by the schema's field order and not by which ones
were right.
"""

import json

import pytest

from seer_annotator.config import ExperimentRun, Question, RunConfig
from seer_annotator.orchestrator import _parse_save_post_tail
from seer_annotator.reply_quality import looks_degenerate, pass2_incomplete_reason
from seer_annotator.store import Store


def _q(vid, key, qtype="text"):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type=qtype,
    )


def _p1_text(*keys):
    return "\n".join(
        f"--- ANSWER: {k} ---\nAnswer: something\nConfidence: 10\n" for k in keys
    )


def _clean_item(key, value="Yes"):
    return {
        "key": key, "value": value, "cited_text": "a quote",
        "comment": "reasoning", "confidence": 18, "status": "ok",
    }


# The prod shape, after json_repair: two finished items, then one whose comment
# ran into a loop, so `confidence` and `status` were never written.
_LOOPED_TAIL = "The study runs several named models (GPT-2 medium, Mistral " + "our " * 400

_REPAIRED_DOC = json.dumps({"results": [
    _clean_item("a"), _clean_item("b", "No"),
    {
        "key": "c", "value": "Yes", "cited_text": "a quote",
        "comment": _LOOPED_TAIL,
    },
]})


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


# --- what counts as unfinished ---------------------------------------------

def test_a_loop_is_detected_from_the_reply_alone():
    code, detail = pass2_incomplete_reason(_REPAIRED_DOC, {})
    assert code == "pass2_degenerate_output"
    # The advice has to contradict the truncation advice, not repeat it.
    assert "does not help" in detail


def test_a_reported_truncation_is_detected():
    code, _ = pass2_incomplete_reason('{"results": [', {"finish_reason": "length"})
    assert code == "pass2_truncated"


def test_a_silent_provider_is_caught_by_the_spent_budget():
    code, _ = pass2_incomplete_reason(
        '{"results": [', {"output_tokens": 8000, "max_tokens": 8000},
    )
    assert code == "pass2_truncated"


def test_a_finished_reply_passes():
    usage = {"finish_reason": "stop", "output_tokens": 120, "max_tokens": 8000}
    assert pass2_incomplete_reason(json.dumps({"results": [_clean_item("a")]}), usage) is None


def test_a_short_reply_with_no_usage_passes():
    """No evidence is not evidence of a fault — the gate must fail open here.

    Batch replies carry a finish reason but no `max_tokens`, and older runs
    carry neither. Failing those would fail every cell in them.
    """
    assert pass2_incomplete_reason(json.dumps({"results": [_clean_item("a")]}), {}) is None
    assert pass2_incomplete_reason(json.dumps({"results": [_clean_item("a")]}), None) is None


def test_a_spent_budget_with_a_normal_finish_reason_passes():
    """`output_tokens == max_tokens` alone is not a truncation.

    It is also what a loop looks like, and what a reply that happened to end on
    the ceiling looks like. The rule only fires where the provider said nothing.
    """
    assert pass2_incomplete_reason(
        json.dumps({"results": [_clean_item("a")]}),
        {"finish_reason": "stop", "output_tokens": 8000, "max_tokens": 8000},
    ) is None


def test_looks_degenerate_ignores_ordinary_json():
    assert not looks_degenerate(json.dumps({"results": [_clean_item("a")]}))
    assert not looks_degenerate("")
    assert not looks_degenerate(None)


# --- what the run does with it ----------------------------------------------

def _run_tail(store, p2_text, p2_usage, keys=("a", "b", "c")):
    group = [_q(i + 1, k) for i, k in enumerate(keys)]
    run = ExperimentRun(
        run_id=1, name="r", model_name="m", model_provider="anthropic",
        config=RunConfig(batching="all"),
    )
    pending = {"cid1": (type("P", (), {"paper_id": 9450})(), group, 0)}
    err = _parse_save_post_tail(
        p1_texts={"cid1": _p1_text(*keys)}, p1_usage={},
        p2_texts={"cid1": p2_text}, p2_usage={"cid1": p2_usage},
        pending_cells=pending, source_texts={9450: "a quote"},
        run=run, cfg=run.config, store=store, fail_fast=False,
    )
    qmap = {i + 1: k for i, k in enumerate(keys)}
    answers = {
        qmap[row["version_id"]]: json.loads(row["payload_json"])
        for row in store.all_answers(run_id=1, paper_id=9450)
    }
    return err, answers


def test_a_looped_reply_fails_every_answer_in_the_group(store):
    """Including the two that parsed cleanly — they are not evidence of quality.

    Which items survive a loop is decided by where the model was in the schema
    when it stopped making sense, so "this one parsed" says only that it came
    first.
    """
    _, answers = _run_tail(store, _REPAIRED_DOC, {})

    assert set(answers) == {"a", "b", "c"}
    assert [answers[k]["extraction_status"] for k in ("a", "b", "c")] == ["error"] * 3
    assert all(answers[k]["value_text"] is None for k in answers)


def test_the_reply_is_kept_so_the_failure_can_be_explained(store):
    """SEER's trace page reads `pass2_text` back to name the loop.

    Failing the cell without keeping the reply would turn a diagnosable failure
    into an unexplained one.
    """
    _, answers = _run_tail(store, _REPAIRED_DOC, {})

    raw = answers["a"]["raw_response"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    assert raw["pass2_text"] == _REPAIRED_DOC
    assert raw["diagnostics"][0]["code"] == "pass2_degenerate_output"
    assert raw["diagnostics"][0]["phase"] == "pass2"


def test_a_truncated_reply_fails_the_group_by_its_own_name(store):
    """A cut-off reply that still parsed. Same outcome, different code.

    Storing the loop's code here would send the reader to raise a ceiling that
    is not the problem, or the reverse.
    """
    doc = json.dumps({"results": [_clean_item("a"), _clean_item("b", "No")]})
    _, answers = _run_tail(store, doc, {"finish_reason": "length"}, keys=("a", "b"))

    assert [answers[k]["extraction_status"] for k in ("a", "b")] == ["error"] * 2
    raw = answers["a"]["raw_response"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    assert raw["diagnostics"][0]["code"] == "pass2_truncated"


def test_a_finished_reply_is_still_saved_normally(store):
    """The gate must not cost anything on a healthy group."""
    doc = json.dumps({"results": [_clean_item("a"), _clean_item("b", "No")]})
    _, answers = _run_tail(
        store, doc,
        {"finish_reason": "stop", "output_tokens": 120, "max_tokens": 8000},
        keys=("a", "b"),
    )

    assert [answers[k]["extraction_status"] for k in ("a", "b")] == ["ok"] * 2
    assert answers["a"]["value_text"] == "Yes"
    assert answers["b"]["value_text"] == "No"


# --- the backstop: an item the reply-level evidence cannot flag --------------
#
# Prod trace 31211: the reply stops mid-word — no closing brace, no loop, and
# `output_tokens` recorded as 0 with no finish reason, so nothing about the
# reply itself says it was cut off. The parser is the only thing left that can
# tell, because json_repair had to close the document to read it at all.

_CUT_OFF_DOC = (
    '{\n  "results": [\n'
    '    {"key": "a", "value": "Yes", "cited_text": "a quote", "comment": "done",'
    ' "confidence": 18, "status": "ok"},\n'
    '    {"key": "b", "value": "No", "cited_text": "a quote", "comment": "the model '
    'was still writing this sentence when it'
)


def test_the_cut_off_item_is_flagged_and_the_finished_one_is_not():
    from seer_annotator.annotate.parse import parse_structured_output

    parsed = {
        r["key"]: r for r in parse_structured_output(
            _CUT_OFF_DOC, ["a", "b"], annotate_mode=True, require_status=True,
        )
    }
    assert "item_error" not in parsed["a"]
    assert "confidence" in parsed["b"]["item_error"]
    assert "status" in parsed["b"]["item_error"]


def test_the_reply_gate_does_not_catch_this_one():
    """Which is the whole reason the parser check exists."""
    assert pass2_incomplete_reason(_CUT_OFF_DOC, {"output_tokens": 0}) is None


def test_a_cut_off_item_becomes_an_error_answer(store):
    _, answers = _run_tail(store, _CUT_OFF_DOC, {"output_tokens": 0}, keys=("a", "b"))

    assert answers["a"]["extraction_status"] == "ok"
    assert answers["b"]["extraction_status"] == "error"
    assert answers["b"]["value_text"] is None


def test_a_clean_document_missing_a_field_is_left_alone():
    """Only salvage is judged this way.

    A document that parsed as written is what the model meant to send, even
    where it skipped a field. Failing those would fail answers on schema
    sloppiness rather than on damage.
    """
    from seer_annotator.annotate.parse import parse_structured_output

    doc = json.dumps({"results": [{"key": "a", "value": "Yes", "cited_text": "q"}]})
    parsed = parse_structured_output(
        doc, ["a"], annotate_mode=True, require_status=True,
    )
    assert "item_error" not in parsed[0]
    assert parsed[0]["status"] == "ok"


def test_arbitration_items_are_not_failed_for_having_no_status():
    """Arbitration asks for no status field, so an item without one is complete."""
    from seer_annotator.annotate.parse import parse_structured_output

    doc = (
        '{"results": [{"key": "a", "value": "Yes", "cited_text": "q", '
        '"comment": "c", "confidence": 18}'
    )  # unterminated: needs repair
    parsed = parse_structured_output(doc, ["a"], annotate_mode=True)
    assert "item_error" not in parsed[0]
