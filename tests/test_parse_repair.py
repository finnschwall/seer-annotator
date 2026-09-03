"""Tests for the whole-document json_repair fallback (parse.py) added to handle
Pass-2 responses that are pretty-printed multi-line JSON missing only the final
closing brace — a shape the pre-existing line-based fallback (JSONL-only) can't
recover at all, previously causing every question to silently placeholder as
status="absent" while the run was still recorded as successful.

Also covers the related "null is equivalent to empty string for cited_text/comment"
fix in parse_structured_output_diagnostic's strict field validation.
"""

import json

from seer_annotator.annotate.parse import (
    parse_format_output,
    parse_structured_output,
    parse_structured_output_diagnostic,
)

# Pretty-printed, multi-line, valid except for the missing final "}" that would
# close the top-level object — mirrors the deepseek-v4-flash failure mode.
_TRUNCATED_DOC = """{"results": [
  {
    "key": "is_paper",
    "value": "Yes",
    "cited_text": "This is a study",
    "comment": "clearly a paper",
    "confidence": 18,
    "status": "ok"
  },
  {
    "key": "n_patients",
    "value": 42,
    "cited_text": "",
    "comment": "",
    "confidence": 15,
    "status": "ok"
  },
  {
    "key": "design",
    "value": "RCT",
    "cited_text": "randomized controlled trial",
    "comment": "stated in methods",
    "confidence": 20,
    "status": "ok"
  },
  {
    "key": "outcome",
    "value": null,
    "cited_text": "",
    "comment": "",
    "confidence": null,
    "status": "absent"
  }
]
"""

_KEYS = ["is_paper", "n_patients", "design", "outcome"]


# ---------------------------------------------------------------------------
# parse_structured_output — whole-document repair
# ---------------------------------------------------------------------------

def test_truncated_document_recovers_all_keys_via_parse_structured_output():
    out = parse_structured_output(_TRUNCATED_DOC, _KEYS, annotate_mode=True)
    by_key = {r["key"]: r for r in out}

    assert by_key["is_paper"]["value"] == "Yes"
    assert by_key["is_paper"]["status"] == "ok"
    assert by_key["n_patients"]["value"] == 42
    assert by_key["design"]["value"] == "RCT"
    assert by_key["design"]["cited_text"] == "randomized controlled trial"
    assert by_key["outcome"]["status"] == "absent"
    assert by_key["outcome"]["value"] is None

    # None of the four keys were fabricated placeholders — every one came from
    # the repaired document, not from _fill_missing's "key not found" path.
    for r in out:
        assert "parse_error" not in r


# ---------------------------------------------------------------------------
# parse_structured_output_diagnostic — whole-document repair + diagnostics
# ---------------------------------------------------------------------------

def test_truncated_document_diagnostic_sets_repair_flags_and_recovers_keys():
    result = parse_structured_output_diagnostic(_TRUNCATED_DOC, _KEYS, annotate_mode=True)

    assert result["native_json"] is False
    assert result["fallback_used"] is True
    assert result["repair_used"] is True
    assert result["missing_keys"] == []
    assert result["duplicate_keys"] == []
    assert result["unexpected_keys"] == []
    assert result["schema_violations"] == []
    assert result["missing_fields"] == []

    by_key = {a["key"]: a for a in result["answers"]}
    assert by_key["is_paper"]["value"] == "Yes"
    assert by_key["is_paper"]["present"] is True
    assert by_key["n_patients"]["value"] == 42
    assert by_key["design"]["cited_text"] == "randomized controlled trial"
    assert by_key["outcome"]["status"] == "absent"


# ---------------------------------------------------------------------------
# null cited_text / comment on an "absent" entry: no schema violation
# ---------------------------------------------------------------------------

def test_null_cited_text_and_comment_on_absent_entry_no_violation():
    text = json.dumps({"results": [
        {"key": "outcome", "value": None, "cited_text": None, "comment": None,
         "confidence": None, "status": "absent"},
    ]})
    result = parse_structured_output_diagnostic(text, ["outcome"], annotate_mode=True)

    assert result["native_json"] is True
    assert result["schema_violations"] == []

    answer = result["answers"][0]
    # Normalized identically to a model that wrote "" instead of null.
    assert answer["cited_text"] == ""
    assert answer["comment"] == ""
    assert answer["status"] == "absent"


def test_empty_string_cited_text_and_comment_still_fine_too():
    """Sanity check: the "" convention (pre-existing) still passes alongside null."""
    text = json.dumps({"results": [
        {"key": "outcome", "value": None, "cited_text": "", "comment": "",
         "confidence": None, "status": "absent"},
    ]})
    result = parse_structured_output_diagnostic(text, ["outcome"], annotate_mode=True)
    assert result["schema_violations"] == []
    assert result["answers"][0]["cited_text"] == ""
    assert result["answers"][0]["comment"] == ""


# ---------------------------------------------------------------------------
# Genuinely unrecoverable response: degrade safely, no exception
# ---------------------------------------------------------------------------

def test_unrecoverable_garbage_degrades_to_placeholders_without_exception():
    garbage = "this is not JSON at all, just prose the model emitted by mistake."

    out = parse_structured_output(garbage, _KEYS, annotate_mode=True)
    assert len(out) == len(_KEYS)
    for r in out:
        assert r["status"] == "absent"
        assert r["value"] is None

    result = parse_structured_output_diagnostic(garbage, _KEYS, annotate_mode=True)
    assert result["native_json"] is False
    assert result["fallback_used"] is True
    assert result["repair_used"] is False
    assert result["missing_keys"] == _KEYS
    for a in result["answers"]:
        assert a["status"] == "absent"
        assert a["present"] is False


# ---------------------------------------------------------------------------
# Existing JSONL-style line-based fallback: no regression
#
# parse_format_output (and the per-line _try_repair it uses) is the actual unit
# under test here — exercised directly, independent of whichever upstream
# caller's whole-document repair attempt may or may not intercept the text
# first (json_repair is aggressive enough to also recover some of these; see
# test_jsonl_fallback_still_works_via_diagnostic below).
# ---------------------------------------------------------------------------

def test_parse_format_output_handles_clean_jsonl_directly():
    text = (
        '{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": 10, "status": "ok"}\n'
        '{"key": "b", "value": 2, "cited_text": "", "comment": "", "confidence": null, "status": "ok"}\n'
    )
    out = parse_format_output(text, ["a", "b"], annotate_mode=True)
    by_key = {r["key"]: r for r in out}
    assert by_key["a"]["value"] == "x"
    assert by_key["b"]["value"] == 2


def test_parse_format_output_still_repairs_single_malformed_line():
    # Trailing comma makes this line invalid JSON on its own; _try_repair's
    # value cross-check (numeric value visible via regex) should accept the fix.
    text = '{"key": "a", "value": 5, "cited_text": "", "comment": "", "confidence": 10,}\n'
    out = parse_format_output(text, ["a"], annotate_mode=True)
    assert out[0]["value"] == 5
    assert out[0]["status"] == "ok"


def test_jsonl_fallback_still_works_via_parse_structured_output():
    # Not valid as a single JSON document (no wrapping {"results": [...]} or
    # [...]), but each line IS a complete JSON object — the pre-existing
    # line-based path this repair must not break.
    text = (
        '{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": 10, "status": "ok"}\n'
        '{"key": "b", "value": 2, "cited_text": "", "comment": "", "confidence": null, "status": "ok"}\n'
    )
    out = parse_structured_output(text, ["a", "b"], annotate_mode=True)
    by_key = {r["key"]: r for r in out}
    assert by_key["a"]["value"] == "x"
    assert by_key["b"]["value"] == 2


def test_jsonl_fallback_still_works_via_diagnostic():
    text = (
        '{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": 10, "status": "ok"}\n'
        '{"key": "b", "value": 2, "cited_text": "", "comment": "", "confidence": null, "status": "ok"}\n'
    )
    result = parse_structured_output_diagnostic(text, ["a", "b"], annotate_mode=True)
    assert result["native_json"] is False
    assert result["fallback_used"] is True
    # This particular JSONL text also happens to be recoverable by the whole-document
    # repair (json_repair merges the two lines into an array before the line-based
    # parser even runs) — either recovery path is acceptable here; what matters is
    # both keys come through correctly either way.
    assert result["missing_keys"] == []
    by_key = {a["key"]: a for a in result["answers"]}
    assert by_key["a"]["value"] == "x"
    assert by_key["b"]["value"] == 2
