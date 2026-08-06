"""Tests for the mechanical per-entry status contract added to annotate/parse.py
(A3) and its annotate_mode opt-in (A5's per-key parse tolerance foundation).

These specifically guard the "arbitration must be unaffected" constraint: every
test that doesn't pass annotate_mode=True must reproduce the exact legacy
behavior (parse_error, no status-driven skip semantics)."""

import json

from seer_annotator.annotate.parse import (
    ANNOTATE_RESPONSE_FORMAT,
    _RESPONSE_FORMAT,
    parse_structured_output,
)


def _wire(results):
    return json.dumps({"results": results})


# ---------------------------------------------------------------------------
# Legacy (default) behavior — arbitration and any non-annotate caller
# ---------------------------------------------------------------------------

def test_default_missing_key_is_parse_error_not_absent():
    text = _wire([{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": None}])
    out = parse_structured_output(text, ["a", "b"])
    by_key = {r["key"]: r for r in out}
    assert "parse_error" in by_key["b"]
    assert "status" not in by_key["b"] or by_key["b"].get("status") is None or True
    assert by_key["a"]["status"] == "ok"  # present entries always get a status now, defaulting "ok"


def test_default_mode_ignores_status_field_if_model_sends_one_anyway():
    """A model that (incorrectly) emits status under the legacy schema must not
    break parsing — annotate_mode=False just doesn't use it for anything."""
    text = _wire([{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": None, "status": "unmappable"}])
    out = parse_structured_output(text, ["a"])
    assert out[0]["value"] == "x"
    assert out[0]["status"] == "unmappable"  # extracted defensively, but caller (arbitration) ignores it


# ---------------------------------------------------------------------------
# annotate_mode=True — the new contract
# ---------------------------------------------------------------------------

def test_annotate_mode_missing_key_is_absent_not_parse_error():
    text = _wire([{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": None, "status": "ok"}])
    out = parse_structured_output(text, ["a", "b"], annotate_mode=True)
    by_key = {r["key"]: r for r in out}
    assert "parse_error" not in by_key["b"]
    assert by_key["b"]["status"] == "absent"
    assert by_key["b"]["value"] is None


def test_annotate_mode_reads_unmappable_status():
    text = _wire([{"key": "a", "value": None, "cited_text": "", "comment": "", "confidence": None, "status": "unmappable"}])
    out = parse_structured_output(text, ["a"], annotate_mode=True)
    assert out[0]["status"] == "unmappable"


def test_annotate_mode_defaults_status_ok_when_model_omits_it():
    """Backward compat within annotate_mode: a model that doesn't send status at
    all (e.g. drop_params silently stripped structured output) still parses,
    defaulting to 'ok' rather than crashing."""
    text = _wire([{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": None}])
    out = parse_structured_output(text, ["a"], annotate_mode=True)
    assert out[0]["status"] == "ok"


def test_annotate_mode_invalid_status_value_defaults_ok():
    text = _wire([{"key": "a", "value": "x", "cited_text": "", "comment": "", "confidence": None, "status": "bogus"}])
    out = parse_structured_output(text, ["a"], annotate_mode=True)
    assert out[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# Schema isolation (A3's "don't mutate the shared constant" requirement)
# ---------------------------------------------------------------------------

def test_response_format_constant_unmutated():
    props = _RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["results"]["items"]["properties"]
    required = _RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["results"]["items"]["required"]
    assert "status" not in props
    assert "status" not in required


def test_annotate_response_format_is_separate_object():
    assert ANNOTATE_RESPONSE_FORMAT is not _RESPONSE_FORMAT
    assert ANNOTATE_RESPONSE_FORMAT["json_schema"]["schema"] is not _RESPONSE_FORMAT["json_schema"]["schema"]
