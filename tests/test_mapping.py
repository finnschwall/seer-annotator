"""Tests for value mapping by question type."""

from decimal import Decimal
import pytest
from seer_annotator.config import Question, QuestionOption
from seer_annotator.mapping import (
    build_llm_answer,
    build_error_answer,
    build_resolution,
    build_error_resolution,
    _map_value,
)


def make_q(qt, allow_multiple=False, options=None):
    return Question(
        question_id=1, key="k", version=1, version_id=1,
        label="L", help_text="", question_type=qt,
        allow_multiple=allow_multiple,
        options=options or [],
    )


OPTS = [QuestionOption(value="rct", label="RCT"), QuestionOption(value="obs", label="Obs")]


def test_boolean_true():
    fields, status, detail = _map_value(make_q("boolean"), True)
    assert fields == {"value_boolean": True}
    assert status == "ok"

def test_boolean_string():
    fields, status, _ = _map_value(make_q("boolean"), "yes")
    assert fields == {"value_boolean": True} and status == "ok"
    fields, status, _ = _map_value(make_q("boolean"), "false")
    assert fields == {"value_boolean": False} and status == "ok"

def test_boolean_invalid_string():
    fields, status, detail = _map_value(make_q("boolean"), "maybe")
    assert fields == {"value_boolean": None}
    assert status == "invalid"
    assert detail == "maybe"

def test_categorical_single():
    q = make_q("categorical", options=OPTS)
    fields, status, _ = _map_value(q, "rct")
    assert fields == {"value_categorical": "rct"} and status == "ok"

def test_categorical_invalid():
    q = make_q("categorical", options=OPTS)
    fields, status, detail = _map_value(q, "garbage")
    assert fields == {"value_categorical": None}
    assert status == "invalid"
    assert detail == "garbage"

def test_categorical_multi():
    q = make_q("categorical", allow_multiple=True, options=OPTS)
    fields, status, _ = _map_value(q, ["rct", "obs"])
    assert fields == {"value_categorical_multi": ["rct", "obs"]} and status == "ok"

def test_categorical_multi_filters_invalid():
    q = make_q("categorical", allow_multiple=True, options=OPTS)
    fields, status, detail = _map_value(q, ["rct", "bad"])
    assert fields == {"value_categorical_multi": ["rct"]}
    assert status == "invalid"
    assert "bad" in detail

def test_text():
    fields, status, _ = _map_value(make_q("text"), "hello")
    assert fields == {"value_text": "hello"} and status == "ok"

def test_integer():
    fields, status, _ = _map_value(make_q("integer"), 42)
    assert fields == {"value_text": "42"} and status == "ok"

def test_float():
    fields, status, _ = _map_value(make_q("float"), 3.14)
    assert fields == {"value_text": "3.14"} and status == "ok"

def test_build_llm_answer_shape():
    q = make_q("text")
    payload = build_llm_answer(
        run_id=1, paper_id=2, question=q, value="hello", comment="c",
        cited_text="span", raw_response={}, latency_ms=100,
        tokens_total=10, tokens_input=8, tokens_output=2, tokens_cached=0,
        cost=Decimal("0.001"),
    )
    assert payload["run"] == 1
    assert payload["paper"] == 2
    assert payload["value_text"] == "hello"
    assert payload["value_boolean"] is None
    assert payload["value_categorical"] is None
    assert payload["value_categorical_multi"] == []
    assert payload["cost"] == "0.001"
    assert payload["cost_currency"] == "USD"
    assert payload["cited_text_verified"] is None
    assert payload["extraction_status"] == "ok"
    assert payload["extraction_detail"] == ""
    assert payload["confidence"] is None


def test_build_llm_answer_invalid_value():
    q = make_q("categorical", options=OPTS)
    payload = build_llm_answer(
        run_id=1, paper_id=2, question=q, value="not_a_valid_option", comment="",
        cited_text="", raw_response={}, latency_ms=0,
        tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
    )
    assert payload["extraction_status"] == "invalid"
    assert payload["extraction_detail"] == "not_a_valid_option"
    assert payload["value_categorical"] is None


def test_build_error_answer():
    q = make_q("categorical", options=OPTS)
    payload = build_error_answer(run_id=1, paper_id=2, question=q, extraction_detail="no_ocr")
    assert payload["extraction_status"] == "error"
    assert payload["extraction_detail"] == "no_ocr"
    assert payload["value_categorical"] is None
    assert payload["confidence"] is None


def test_build_llm_answer_confidence():
    q = make_q("text")
    payload = build_llm_answer(
        run_id=1, paper_id=2, question=q, value="x", comment="", cited_text="",
        raw_response={}, latency_ms=0,
        tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
        confidence=18,
    )
    assert payload["confidence"] == 18


def test_build_llm_answer_cited_text_verified():
    q = make_q("text")
    payload_true = build_llm_answer(
        run_id=1, paper_id=2, question=q, value="hello", comment="c",
        cited_text="span", cited_text_verified=True, raw_response={}, latency_ms=0,
        tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
    )
    assert payload_true["cited_text_verified"] is True

    payload_false = build_llm_answer(
        run_id=1, paper_id=2, question=q, value="hello", comment="c",
        cited_text="span", cited_text_verified=False, raw_response={}, latency_ms=0,
        tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
    )
    assert payload_false["cited_text_verified"] is False


def test_build_resolution_shape():
    q = make_q("categorical", options=OPTS)
    payload = build_resolution(
        arbiter_run_id=21, paper_id=42, dispute_item_id=501, question=q,
        value="rct", comment="c", cited_text="span", raw_response={}, latency_ms=100,
        tokens_total=10, tokens_input=8, tokens_output=2, tokens_cached=0,
        cost=Decimal("0.001"), confidence=18,
    )
    assert payload["arbiter_run"] == 21
    assert payload["paper"] == 42
    assert payload["dispute_item"] == 501
    assert payload["question_version"] == q.version_id
    assert payload["value_categorical"] == "rct"
    assert payload["resolution_status"] == "ok"
    assert payload["resolution_detail"] == ""
    assert payload["confidence"] == 18
    assert payload["cost"] == "0.001"
    # Resolution write API has no cited_text_verified field — computed locally only.
    assert "cited_text_verified" in payload
    assert payload["cited_text_verified"] is None


def test_build_resolution_invalid_value():
    q = make_q("categorical", options=OPTS)
    payload = build_resolution(
        arbiter_run_id=21, paper_id=42, dispute_item_id=501, question=q,
        value="not_a_valid_option", comment="", cited_text="", raw_response={},
        latency_ms=0, tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
    )
    assert payload["resolution_status"] == "invalid"
    assert payload["resolution_detail"] == "not_a_valid_option"
    assert payload["value_categorical"] is None


def test_build_resolution_null_value_is_abstention_not_error():
    """A null value is a legitimate abstention (matches LLMAnswer's convention),
    not an error — resolution_status stays 'ok' with an explanatory detail."""
    q = make_q("categorical", options=OPTS)
    payload = build_resolution(
        arbiter_run_id=21, paper_id=42, dispute_item_id=501, question=q,
        value=None, comment="", cited_text="", raw_response={},
        latency_ms=0, tokens_total=0, tokens_input=0, tokens_output=0, tokens_cached=0, cost=None,
    )
    assert payload["resolution_status"] == "ok"
    assert payload["resolution_detail"] == "LLM answered: non-determinable"
    assert payload["value_categorical"] is None


def test_build_error_resolution():
    q = make_q("categorical", options=OPTS)
    payload = build_error_resolution(
        arbiter_run_id=21, paper_id=42, dispute_item_id=501, question=q,
        resolution_detail="no_ocr",
    )
    assert payload["resolution_status"] == "error"
    assert payload["resolution_detail"] == "no_ocr"
    assert payload["value_categorical"] is None
    assert payload["confidence"] is None
