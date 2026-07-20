"""Map internal answer values to LLMAnswer payload fields by question type."""

from __future__ import annotations

from decimal import Decimal

from .config import Question


_NULL_DETAIL = "LLM answered: non-determinable"

_EMPTY_PAYLOAD = {
    "value_text": None,
    "value_boolean": None,
    "value_categorical": None,
    "value_categorical_multi": [],
    "extraction_status": "ok",
    "extraction_detail": "",
    "confidence": None,
}


def build_llm_answer(
    *,
    run_id: int,
    paper_id: int,
    question: Question,
    value: object,
    comment: str,
    cited_text: str,
    cited_text_verified: bool | None = None,
    citations: list | None = None,
    raw_response: dict,
    latency_ms: int,
    tokens_total: int,
    tokens_input: int,
    tokens_output: int,
    tokens_cached: int,
    cost: Decimal | None,
    cost_currency: str = "USD",
    fmt_tokens_total: int = 0,
    fmt_tokens_input: int = 0,
    fmt_tokens_output: int = 0,
    fmt_tokens_cached: int = 0,
    fmt_cost: Decimal | None = None,
    confidence: int | None = None,
    extraction_status: str | None = None,
    extraction_detail: str | None = None,
) -> dict:
    payload = dict(_EMPTY_PAYLOAD)
    field_dict, derived_status, derived_detail = _map_value(question, value)
    payload.update(field_dict)
    payload["extraction_status"] = extraction_status if extraction_status is not None else derived_status
    payload["extraction_detail"] = extraction_detail if extraction_detail is not None else derived_detail
    payload["confidence"] = confidence

    import json

    payload.update(
        {
            "run": run_id,
            "paper": paper_id,
            "question_version": question.version_id,
            "comment": comment,
            "cited_text": cited_text,
            "cited_text_verified": cited_text_verified,
            "citations": citations or [],
            "raw_response": json.dumps(raw_response, default=str),
            "latency_ms": latency_ms,
            "tokens_total": tokens_total,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_cached": tokens_cached,
            "cost": str(cost) if cost is not None else None,
            "cost_currency": cost_currency,
            "fmt_tokens_total": fmt_tokens_total,
            "fmt_tokens_input": fmt_tokens_input,
            "fmt_tokens_output": fmt_tokens_output,
            "fmt_tokens_cached": fmt_tokens_cached,
            "fmt_cost": str(fmt_cost) if fmt_cost is not None else None,
        }
    )
    return payload


_EMPTY_RESOLUTION_PAYLOAD = {
    "value_text": None,
    "value_boolean": None,
    "value_categorical": None,
    "value_categorical_multi": [],
    "resolution_status": "ok",
    "resolution_detail": "",
    "confidence": None,
}


def build_resolution(
    *,
    arbiter_run_id: int,
    paper_id: int,
    dispute_item_id: int,
    question: Question,
    value: object,
    comment: str,
    cited_text: str,
    cited_text_verified: bool | None = None,
    raw_response: dict,
    latency_ms: int,
    tokens_total: int,
    tokens_input: int,
    tokens_output: int,
    tokens_cached: int,
    cost: Decimal | None,
    cost_currency: str = "USD",
    fmt_tokens_total: int = 0,
    fmt_tokens_input: int = 0,
    fmt_tokens_output: int = 0,
    fmt_tokens_cached: int = 0,
    fmt_cost: Decimal | None = None,
    confidence: int | None = None,
    resolution_status: str | None = None,
    resolution_detail: str | None = None,
) -> dict:
    """Build a Resolution payload — mirrors build_llm_answer() for the arbitration path.

    Reuses _map_value() unchanged: its "ok" + null-value-detail behavior already
    matches the adjudication contract's abstention semantics (a null value with an
    explanatory detail, not an error). cited_text_verified is computed and stored
    locally for debugging only — the Resolution write API has no such field, unlike
    LLMAnswer.
    """
    payload = dict(_EMPTY_RESOLUTION_PAYLOAD)
    field_dict, derived_status, derived_detail = _map_value(question, value)
    payload.update(field_dict)
    payload["resolution_status"] = resolution_status if resolution_status is not None else derived_status
    payload["resolution_detail"] = resolution_detail if resolution_detail is not None else derived_detail
    payload["confidence"] = confidence

    import json

    payload.update(
        {
            "arbiter_run": arbiter_run_id,
            "paper": paper_id,
            "dispute_item": dispute_item_id,
            "question_version": question.version_id,
            "comment": comment,
            "cited_text": cited_text,
            "cited_text_verified": cited_text_verified,
            "raw_response": json.dumps(raw_response, default=str),
            "latency_ms": latency_ms,
            "tokens_total": tokens_total,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_cached": tokens_cached,
            "cost": str(cost) if cost is not None else None,
            "cost_currency": cost_currency,
            "fmt_tokens_total": fmt_tokens_total,
            "fmt_tokens_input": fmt_tokens_input,
            "fmt_tokens_output": fmt_tokens_output,
            "fmt_tokens_cached": fmt_tokens_cached,
            "fmt_cost": str(fmt_cost) if fmt_cost is not None else None,
        }
    )
    return payload


def build_error_answer(
    *,
    run_id: int,
    paper_id: int,
    question: Question,
    extraction_detail: str,
) -> dict:
    """Build a minimal error payload to post to SEER when the pipeline fails for a question."""
    return build_llm_answer(
        run_id=run_id,
        paper_id=paper_id,
        question=question,
        value=None,
        comment="",
        cited_text="",
        cited_text_verified=None,
        citations=[],
        raw_response={},
        latency_ms=0,
        tokens_total=0,
        tokens_input=0,
        tokens_output=0,
        tokens_cached=0,
        cost=None,
        extraction_status="error",
        extraction_detail=extraction_detail,
    )


def build_error_resolution(
    *,
    arbiter_run_id: int,
    paper_id: int,
    dispute_item_id: int,
    question: Question,
    resolution_detail: str,
) -> dict:
    """Build a minimal error Resolution payload when adjudication fails for a dispute."""
    return build_resolution(
        arbiter_run_id=arbiter_run_id,
        paper_id=paper_id,
        dispute_item_id=dispute_item_id,
        question=question,
        value=None,
        comment="",
        cited_text="",
        cited_text_verified=None,
        raw_response={},
        latency_ms=0,
        tokens_total=0,
        tokens_input=0,
        tokens_output=0,
        tokens_cached=0,
        cost=None,
        resolution_status="error",
        resolution_detail=resolution_detail,
    )


def _map_value(question: Question, value: object) -> tuple[dict, str, str]:
    """Returns (field_dict, extraction_status, extraction_detail)."""
    qt = question.question_type

    if qt == "boolean":
        if isinstance(value, bool):
            return {"value_boolean": value}, "ok", ""
        if value is None:
            return {"value_boolean": None}, "ok", _NULL_DETAIL
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "yes", "1"):
                return {"value_boolean": True}, "ok", ""
            if lower in ("false", "no", "0"):
                return {"value_boolean": False}, "ok", ""
            return {"value_boolean": None}, "invalid", value
        return {"value_boolean": bool(value)}, "ok", ""

    if qt == "categorical":
        valid = {opt.value for opt in question.options}
        if question.allow_multiple:
            if isinstance(value, list):
                valid_vals = [str(v) for v in value if str(v) in valid]
                invalid_vals = [str(v) for v in value if str(v) not in valid]
                if invalid_vals:
                    return {"value_categorical_multi": valid_vals}, "invalid", str(value)
                return {"value_categorical_multi": valid_vals}, "ok", ""
            else:
                if value is None:
                    return {"value_categorical_multi": []}, "ok", _NULL_DETAIL
                sv = str(value)
                if sv in valid:
                    return {"value_categorical_multi": [sv]}, "ok", ""
                return {"value_categorical_multi": []}, "invalid", sv
        else:
            if value is None:
                return {"value_categorical": None}, "ok", _NULL_DETAIL
            sv = str(value)
            if sv in valid:
                return {"value_categorical": sv}, "ok", ""
            return {"value_categorical": None}, "invalid", sv

    if qt == "text":
        return {"value_text": str(value) if value is not None else None}, "ok", ""

    if qt in ("integer", "float"):
        return {"value_text": str(value) if value is not None else None}, "ok", ""

    raise ValueError(f"Unknown question_type: {qt!r}")
