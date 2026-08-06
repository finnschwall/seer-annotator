"""Tests for annotate/scope.py — the deterministic IC-gate scope enforcement
that's authoritative over the model (A4). Pure logic, no network/DB."""

from seer_annotator.config import Question, QuestionOption
from seer_annotator.annotate.scope import (
    apply_scope_and_status,
    compute_exclusion_index,
    ic_answer_passes,
    option_ic_passes,
)


def make_bool_ic(key, vid, include_when_true=True):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type="boolean", is_ic=True, ic_include_when_true=include_when_true,
    )


def make_cat_ic(key, vid, options):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type="categorical", is_ic=True, options=options,
    )


def make_plain(key, vid, qtype="text"):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type=qtype,
    )


# ---------------------------------------------------------------------------
# ic_answer_passes / option_ic_passes
# ---------------------------------------------------------------------------

def test_boolean_ic_passes_and_fails():
    q = make_bool_ic("model_scope", 1, include_when_true=True)
    assert ic_answer_passes(q, True) is True
    assert ic_answer_passes(q, False) is False


def test_boolean_ic_include_when_false():
    q = make_bool_ic("excl_flag", 1, include_when_true=False)
    assert ic_answer_passes(q, False) is True
    assert ic_answer_passes(q, True) is False


def test_boolean_ic_indeterminate_without_ic_include_when_true():
    q = Question(
        question_id=1, key="k", version=1, version_id=1, label="k",
        question_type="boolean", is_ic=True, ic_include_when_true=None,
    )
    assert ic_answer_passes(q, True) is None


def test_non_ic_question_always_none():
    q = make_plain("age", 1, "text")
    assert ic_answer_passes(q, "anything") is None


def test_categorical_single_ic_passes():
    opts = [
        QuestionOption(value="rct", label="RCT", ic_passes=1),
        QuestionOption(value="case_report", label="Case report", ic_passes=0),
    ]
    q = make_cat_ic("design", 1, opts)
    assert ic_answer_passes(q, "rct") is True
    assert ic_answer_passes(q, "case_report") is False


def test_categorical_unknown_value_is_indeterminate():
    opts = [QuestionOption(value="rct", label="RCT", ic_passes=1)]
    q = make_cat_ic("design", 1, opts)
    assert ic_answer_passes(q, "unknown_value") is None
    assert option_ic_passes(q, "unknown_value") is None


def test_categorical_multi_all_must_pass():
    q = Question(
        question_id=1, key="k", version=1, version_id=1, label="k",
        question_type="categorical", is_ic=True, allow_multiple=True,
        options=[
            QuestionOption(value="a", label="A", ic_passes=1),
            QuestionOption(value="b", label="B", ic_passes=0),
        ],
    )
    assert ic_answer_passes(q, ["a"]) is True
    assert ic_answer_passes(q, ["a", "b"]) is False
    assert ic_answer_passes(q, []) is None


def test_categorical_option_ic_passes_none_when_unset():
    opts = [QuestionOption(value="x", label="X", ic_passes=None)]
    q = make_cat_ic("design", 1, opts)
    assert ic_answer_passes(q, "x") is None


# ---------------------------------------------------------------------------
# compute_exclusion_index
# ---------------------------------------------------------------------------

def test_compute_exclusion_index_first_failure_wins():
    q1 = make_bool_ic("a", 1, include_when_true=True)
    q2 = make_bool_ic("b", 2, include_when_true=True)
    q3 = make_bool_ic("c", 3, include_when_true=True)
    questions = [q1, q2, q3]
    # a passes, b fails, c would also fail but b is first in order
    values = {"a": True, "b": False, "c": False}
    assert compute_exclusion_index(questions, values) == 1


def test_compute_exclusion_index_no_exclusion():
    q1 = make_bool_ic("a", 1, include_when_true=True)
    q2 = make_plain("free_text", 2)
    values = {"a": True, "free_text": "whatever"}
    assert compute_exclusion_index([q1, q2], values) is None


def test_compute_exclusion_index_ignores_non_ic_questions():
    q1 = make_plain("intro", 1)
    q2 = make_bool_ic("gate", 2, include_when_true=True)
    values = {"intro": "anything", "gate": False}
    assert compute_exclusion_index([q1, q2], values) == 1


def test_compute_exclusion_index_missing_value_is_no_signal():
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    # "gate" absent from values entirely (e.g. Pass-2 status != "ok")
    assert compute_exclusion_index([q1], {}) is None


# ---------------------------------------------------------------------------
# apply_scope_and_status
# ---------------------------------------------------------------------------

def _result(key, value, status="ok"):
    return {"key": key, "value": value, "cited_text": "", "comment": "", "confidence": None, "status": status}


def test_apply_scope_marks_downstream_skipped_when_enabled():
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    q2 = make_plain("downstream", 2)
    parsed = [_result("gate", False), _result("downstream", "some answer")]

    out, excl_idx = apply_scope_and_status([q1, q2], parsed, enabled=True)

    assert excl_idx == 0
    assert out[0]["extraction_status"] == "ok"  # the excluding answer itself is real
    assert out[1]["extraction_status"] == "skipped"
    assert out[1]["value"] is None
    assert "gate" in out[1]["extraction_detail"]


def test_apply_scope_downstream_answered_anyway_is_discarded():
    """Even if the model didn't obey the stop instruction and answered past the
    exclusion, the worker discards the moot value (A4 correctness guarantee)."""
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    q2 = make_plain("downstream", 2)
    parsed = [_result("gate", False), _result("downstream", "model answered anyway")]

    out, excl_idx = apply_scope_and_status([q1, q2], parsed, enabled=True)

    assert out[1]["extraction_status"] == "skipped"
    assert out[1]["value"] is None


def test_apply_scope_disabled_never_produces_skipped():
    """enabled=False (RunConfig.early_exit_on_ic_exclusion off) must behave as
    if there's never an exclusion — behavior-preservation guarantee."""
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    q2 = make_plain("downstream", 2)
    parsed = [_result("gate", False), _result("downstream", "some answer")]

    out, excl_idx = apply_scope_and_status([q1, q2], parsed, enabled=False)

    assert excl_idx is None
    assert out[0]["extraction_status"] == "ok"
    assert out[1]["extraction_status"] == "ok"
    assert out[1]["value"] == "some answer"


def test_apply_scope_unmappable_becomes_invalid():
    q1 = make_plain("q", 1)
    parsed = [_result("q", None, status="unmappable")]
    out, excl_idx = apply_scope_and_status([q1], parsed, enabled=True)
    assert excl_idx is None
    assert out[0]["extraction_status"] == "invalid"


def test_apply_scope_absent_stays_absent_when_no_exclusion():
    q1 = make_plain("q", 1)
    parsed = [_result("q", None, status="absent")]
    out, excl_idx = apply_scope_and_status([q1], parsed, enabled=True)
    assert out[0]["extraction_status"] == "absent"


def test_apply_scope_absent_before_exclusion_is_not_skipped():
    """An 'absent' key at/before the exclusion point is a genuine omission,
    not moot — it must NOT be relabeled 'skipped'."""
    q1 = make_plain("early", 1)
    q2 = make_bool_ic("gate", 2, include_when_true=True)
    q3 = make_plain("downstream", 3)
    parsed = [
        _result("early", None, status="absent"),
        _result("gate", False),
        _result("downstream", "answer"),
    ]
    out, excl_idx = apply_scope_and_status([q1, q2, q3], parsed, enabled=True)
    assert excl_idx == 1
    assert out[0]["extraction_status"] == "absent"  # before exclusion point — genuine gap
    assert out[1]["extraction_status"] == "ok"
    assert out[2]["extraction_status"] == "skipped"  # after exclusion point


def test_apply_scope_absent_ic_answer_is_no_exclusion_signal():
    """If the IC answer itself is absent, there's nothing to gate on — later
    questions must NOT be marked skipped."""
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    q2 = make_plain("downstream", 2)
    parsed = [_result("gate", None, status="absent"), _result("downstream", "answer")]
    out, excl_idx = apply_scope_and_status([q1, q2], parsed, enabled=True)
    assert excl_idx is None
    assert out[1]["extraction_status"] == "ok"


def test_apply_scope_does_not_mutate_input():
    q1 = make_bool_ic("gate", 1, include_when_true=True)
    q2 = make_plain("downstream", 2)
    parsed = [_result("gate", False), _result("downstream", "answer")]
    original_downstream = dict(parsed[1])
    apply_scope_and_status([q1, q2], parsed, enabled=True)
    assert parsed[1] == original_downstream
