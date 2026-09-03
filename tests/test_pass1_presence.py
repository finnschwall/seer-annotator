"""Tests for the pass-1 answer-block presence guard: a Pass-2 entry may only
keep a status that claims its answer block exists ("ok"/"unmappable") if
Pass-1's free-form text actually contains an answer block for that question —
an "--- ANSWER: <key> ---" header, or a "--- QUESTION: <key> ---" header over
a block that holds an "Answer:" field. This is a DEMOTE-ONLY correctness guarantee — it never
promotes "absent" -> "ok" — added because a weak local model was observed to
fabricate values (copying another question's quotes/comment, confidence 20)
for questions Pass-1 never answered.

Covers:
  - seer_annotator.annotate.parse.pass1_block_present (the tolerant matcher)
  - seer_annotator.annotate.scope.apply_scope_and_status's pass1_text= guard,
    including the critical ordering guarantee: a fabricated "ok" must not be
    able to move the IC exclusion point.
"""

from seer_annotator.annotate.parse import pass1_block_present
from seer_annotator.annotate.scope import apply_scope_and_status
from seer_annotator.config import Question


# ---------------------------------------------------------------------------
# pass1_block_present
# ---------------------------------------------------------------------------

def test_exact_header_matches():
    text = "--- ANSWER: study_design ---\nAnswer: rct\n"
    assert pass1_block_present(text, "study_design") is True


def test_case_insensitive():
    text = "--- answer: study_design ---\nAnswer: rct\n"
    assert pass1_block_present(text, "study_design") is True


def test_dash_and_asterisk_variants_match():
    assert pass1_block_present("*** ANSWER: q1 ***\nAnswer: x\n", "q1") is True
    assert pass1_block_present("===ANSWER: q1===\nAnswer: x\n", "q1") is True
    assert pass1_block_present("ANSWER:q1\nAnswer: x\n", "q1") is True
    assert pass1_block_present("  ANSWER  :  q1\nAnswer: x\n", "q1") is True


def test_whitespace_variants_match():
    text = "---   ANSWER   :   q1   ---\nAnswer: x\n"
    assert pass1_block_present(text, "q1") is True


def test_q1_does_not_match_text_containing_only_q10():
    text = "--- ANSWER: q10 ---\nAnswer: x\n"
    assert pass1_block_present(text, "q1") is False
    assert pass1_block_present(text, "q10") is True


def test_missing_key_does_not_match():
    text = "--- ANSWER: other_key ---\nAnswer: x\n"
    assert pass1_block_present(text, "study_design") is False


def test_empty_text_does_not_match():
    assert pass1_block_present("", "q1") is False


# ---------------------------------------------------------------------------
# pass1_block_present: block labelled after the QUESTION instead of the ANSWER
#
# Observed in production (run "deepseek-v4-flash-thinking", 11 of 60,826
# question slots): the model answers correctly but heads the block with the
# questions message's "--- QUESTION: <key> ---" shape. Recoverable only when
# the block actually holds an answer — a bare echoed header must stay absent.
# ---------------------------------------------------------------------------

_ANSWERED_BLOCK = """--- QUESTION: xai_operation ---
Quotes:
- "the abstract does not describe any attribution method"
Reasoning: The role-specialised agents perform prediction, not analysis.
Answer: No
Confidence: 18
"""


def test_question_header_with_answer_field_matches():
    assert pass1_block_present(_ANSWERED_BLOCK, "xai_operation") is True


def test_question_header_after_a_real_answer_block_matches():
    text = "--- ANSWER: is_paper ---\nAnswer: Yes\nConfidence: 20\n\n" + _ANSWERED_BLOCK
    assert pass1_block_present(text, "is_paper") is True
    assert pass1_block_present(text, "xai_operation") is True


def test_decorated_answer_field_still_counts_as_content():
    text = "--- QUESTION: q1 ---\nReasoning: because\n**Answer:** Yes\n"
    assert pass1_block_present(text, "q1") is True


def test_bare_question_header_does_not_match():
    """The hole this guard must keep shut: a header echoed with nothing
    answered under it, so Pass-2 has no answer to reformat and any value it
    reports is invented."""
    text = "--- QUESTION: q1 ---\n--- STOP: exclusion criteria met ---\n"
    assert pass1_block_present(text, "q1") is False


def test_question_header_does_not_borrow_the_next_blocks_answer():
    text = "--- QUESTION: q1 ---\n--- ANSWER: q2 ---\nAnswer: Yes\nConfidence: 20\n"
    assert pass1_block_present(text, "q1") is False
    assert pass1_block_present(text, "q2") is True


def test_echoed_question_body_does_not_match():
    """A model restating the prompt's question block verbatim (Label/options,
    no answer) has not answered it."""
    text = (
        "--- QUESTION: study_design ---\n"
        "Label: Study design\n"
        "Valid options (use the 'value' string exactly):\n"
        "  'rct' — Randomised controlled trial\n"
    )
    assert pass1_block_present(text, "study_design") is False


def test_question_fallback_keeps_key_boundaries_exact():
    text = "--- QUESTION: q10 ---\nAnswer: x\n"
    assert pass1_block_present(text, "q1") is False
    assert pass1_block_present(text, "q10") is True


# ---------------------------------------------------------------------------
# apply_scope_and_status(pass1_text=...) demotion
# ---------------------------------------------------------------------------

def _plain(key, vid, qtype="text"):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type=qtype,
    )


def _bool_ic(key, vid, include_when_true=True):
    return Question(
        question_id=vid, key=key, version=1, version_id=vid, label=key,
        question_type="boolean", is_ic=True, ic_include_when_true=include_when_true,
    )


def _result(key, value, status="ok", **extra):
    r = {"key": key, "value": value, "cited_text": "quote", "comment": "reasoning",
         "confidence": 20, "status": status}
    r.update(extra)
    return r


def test_fabricated_ok_is_demoted_to_absent():
    """A Pass-2 'ok' for a key whose block is absent from pass-1 must be
    demoted, with the fabricated payload fully cleared."""
    q = _plain("fabricated_q", 1)
    pass1_text = "--- ANSWER: some_other_q ---\nAnswer: yes\n"
    parsed = [_result("fabricated_q", "invented value")]

    out, _ = apply_scope_and_status([q], parsed, enabled=False, pass1_text=pass1_text)

    entry = out[0]
    assert entry["extraction_status"] == "absent"
    assert entry["value"] is None
    assert entry["cited_text"] == ""
    assert entry["comment"] == ""
    assert entry["confidence"] is None
    assert "pass-1" in entry["extraction_detail"]
    assert "fabricated_q" in entry["extraction_detail"]


def test_fabricated_unmappable_is_demoted_to_absent():
    q = _plain("q1", 1)
    pass1_text = "--- ANSWER: other ---\nAnswer: yes\n"
    parsed = [_result("q1", None, status="unmappable")]

    out, _ = apply_scope_and_status([q], parsed, enabled=False, pass1_text=pass1_text)

    assert out[0]["extraction_status"] == "absent"
    assert out[0]["value"] is None
    assert out[0]["cited_text"] == ""
    assert out[0]["comment"] == ""
    assert out[0]["confidence"] is None


def test_genuine_ok_with_block_present_is_untouched():
    q = _plain("study_design", 1)
    pass1_text = "--- ANSWER: study_design ---\nAnswer: rct\nConfidence: 15\n"
    parsed = [_result("study_design", "rct", confidence=15)]

    out, _ = apply_scope_and_status([q], parsed, enabled=False, pass1_text=pass1_text)

    assert out[0]["extraction_status"] == "ok"
    assert out[0]["value"] == "rct"
    assert out[0]["cited_text"] == "quote"
    assert out[0]["comment"] == "reasoning"
    assert out[0]["confidence"] == 15


def test_pass1_text_none_changes_nothing():
    """Regression guard: pass1_text=None (the default, used by arbitration
    call sites, which never pass this parameter) must perform no check at
    all — a fabricated-looking 'ok' entry survives unchanged."""
    q = _plain("fabricated_q", 1)
    parsed = [_result("fabricated_q", "invented value")]

    out, _ = apply_scope_and_status([q], parsed, enabled=False, pass1_text=None)

    assert out[0]["extraction_status"] == "ok"
    assert out[0]["value"] == "invented value"
    assert out[0]["cited_text"] == "quote"
    assert out[0]["comment"] == "reasoning"
    assert out[0]["confidence"] == 20


def test_fabricated_ok_on_early_ic_question_cannot_move_exclusion_point():
    """The important one: Pass-1 only answered a LATER question. Pass-2
    fabricates 'excluding' values for two EARLIER IC gate questions. Since the
    demotion runs before compute_exclusion_index, those fabricated answers
    must never be allowed to move the exclusion point — the exclusion must
    come only from the IC question Pass-1 actually answered."""
    gate1 = _bool_ic("gate1", 1, include_when_true=True)   # Pass-1 never answers this
    gate2 = _bool_ic("gate2", 2, include_when_true=True)   # Pass-1 never answers this
    real_gate = _bool_ic("real_gate", 3, include_when_true=True)  # Pass-1 DOES answer this
    downstream = _plain("downstream", 4)

    questions = [gate1, gate2, real_gate, downstream]

    # Pass-1 text has an answer block ONLY for real_gate.
    pass1_text = "--- ANSWER: real_gate ---\nAnswer: false\nConfidence: 18\n"

    # Pass-2 fabricates "excluding" (False) values for gate1/gate2 — exactly
    # the failure mode observed with a weak local model: copied-looking
    # values with high confidence, for questions pass-1 never touched.
    parsed = [
        _result("gate1", False),
        _result("gate2", False),
        _result("real_gate", False),
        _result("downstream", "some answer"),
    ]

    out, excl_idx = apply_scope_and_status(
        questions, parsed, enabled=True, pass1_text=pass1_text,
    )

    # The exclusion must be attributed to real_gate (index 2), NOT gate1 (0)
    # or gate2 (1) — those were fabricated and demoted to "absent" before the
    # exclusion index was ever computed.
    assert excl_idx == 2
    assert out[0]["extraction_status"] == "absent"
    assert out[1]["extraction_status"] == "absent"
    assert out[2]["extraction_status"] == "ok"
    assert out[2]["value"] is False
    # downstream, ordered after the real exclusion point, is discarded as moot.
    assert out[3]["extraction_status"] == "skipped"
    assert "real_gate" in out[3]["extraction_detail"]


def test_question_labelled_block_keeps_status_ok():
    """End-to-end shape of the production bug: Pass-1 answered both keys but
    labelled the second block after the question. Neither entry may be
    demoted — the demotion of one key fails the whole group upstream."""
    q1 = _plain("is_paper", 1)
    q2 = _plain("xai_operation", 2)
    pass1_text = (
        "--- ANSWER: is_paper ---\nAnswer: Yes\nConfidence: 20\n\n"
        "--- QUESTION: xai_operation ---\nAnswer: No\nConfidence: 18\n"
    )
    parsed = [_result("is_paper", "Yes"), _result("xai_operation", "No", confidence=18)]

    out, _ = apply_scope_and_status([q1, q2], parsed, enabled=False, pass1_text=pass1_text)

    assert [e["extraction_status"] for e in out] == ["ok", "ok"]
    assert [e["value"] for e in out] == ["Yes", "No"]
