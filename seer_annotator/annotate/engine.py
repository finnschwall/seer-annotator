"""Two-pass annotation engine: reason freely, then format, then verify."""

from __future__ import annotations

import json
import logging
import pathlib
import uuid
from decimal import Decimal
from typing import Callable

logger = logging.getLogger(__name__)

from ..caching import apply_cache
from ..config import DEFAULT_REQUEST_TIMEOUT, ExperimentRun, Question
from ..llm import LLMResult, complete as llm_complete, dummy_complete
from ..mapping import build_error_answer, build_llm_answer, build_skipped_answer
from .citation import build_citations
from .prompt import build_messages, build_format_messages
from .parse import (
    ANNOTATE_RESPONSE_FORMAT,
    ExtractionError,
    parse_structured_output,
    pass1_block_present,
    _RESPONSE_FORMAT,
)
from .scope import apply_scope_and_status
from .verify import verify_citation, verify_citations


CompleteFn = Callable[..., object]  # async (model, provider, messages, **kw) -> LLMResult


def _dump_debug(
    group_id: str,
    p1_text: str,
    p2_text: str,
    parsed: list[dict],
    paper_id: int,
) -> None:
    """Write pass-1, pass-2, and parse results to ./seer-debug/ for post-mortem inspection."""
    debug_dir = pathlib.Path("seer-debug")
    debug_dir.mkdir(exist_ok=True)
    stem = debug_dir / f"paper{paper_id}-{group_id}"
    try:
        (stem.parent / f"{stem.name}-p1.txt").write_text(p1_text)
        (stem.parent / f"{stem.name}-p2.txt").write_text(p2_text)
        (stem.parent / f"{stem.name}-parse.json").write_text(
            json.dumps(parsed, indent=2, default=str)
        )
        logger.error(
            "Pass-2 debug dump written to:\n"
            "  pass-1 output : %s-p1.txt\n"
            "  pass-2 output : %s-p2.txt\n"
            "  parse results : %s-parse.json",
            stem, stem, stem,
        )
    except OSError as exc:
        logger.warning("Could not write debug dump to %s*: %s", stem, exc)


async def annotate_group(
    *,
    run: ExperimentRun,
    paper_id: int,
    source_text: str,
    questions: list[Question],
    format_model: str,
    format_model_provider: str,
    complete_fn: CompleteFn = llm_complete,
    format_complete_fn: CompleteFn | None = None,
    batch_group_id: str | None = None,
    system_prompt: str | None = None,
    citation_max_error_rate: float = 0.05,
    citation_max_ellipsis_gap: int = 300,
) -> list[dict]:
    """Run pass-1 + pass-2 for a group of questions, return list of LLMAnswer payloads."""
    if batch_group_id is None:
        batch_group_id = str(uuid.uuid4())[:8]

    cfg = run.config

    # ---------- Pass 1: free-form reasoning ----------
    messages = build_messages(
        source_text,
        questions,
        text_source=cfg.text_source,
        system_prompt=system_prompt,
        cache_first=cfg.cache_first,
        early_exit_on_ic_exclusion=cfg.early_exit_on_ic_exclusion,
    )
    messages = apply_cache(run.model_provider, messages, cfg.cache)

    p1_kwargs: dict = {
        "timeout": cfg.request_timeout or DEFAULT_REQUEST_TIMEOUT,
    }
    if cfg.temperature is not None:
        p1_kwargs["temperature"] = cfg.temperature
    if cfg.reasoning_effort:
        p1_kwargs["reasoning_effort"] = cfg.reasoning_effort
    p1_kwargs.update(cfg.model_params)

    p1: LLMResult = await complete_fn(  # type: ignore[assignment]
        run.model_name,
        run.model_provider,
        messages,
        **p1_kwargs,
    )

    # ---------- Pass 2: structured formatting ----------
    p2_model = cfg.format_model or format_model
    p2_provider = cfg.format_model_provider or format_model_provider

    fmt_messages = build_format_messages(p1.text, questions, require_status=True)

    p2_kwargs: dict = {
        "timeout": cfg.request_timeout or DEFAULT_REQUEST_TIMEOUT,
    }
    if cfg.format_temperature is not None:
        p2_kwargs["temperature"] = cfg.format_temperature
    p2_kwargs.update(cfg.format_model_params)
    if cfg.format_structured_output:
        p2_kwargs["response_format"] = ANNOTATE_RESPONSE_FORMAT

    p2_fn = format_complete_fn if format_complete_fn is not None else complete_fn
    p2: LLMResult = await p2_fn(  # type: ignore[assignment]
        p2_model,
        p2_provider,
        fmt_messages,
        **p2_kwargs,
    )

    parsed = parse_structured_output(p2.text, [q.key for q in questions], annotate_mode=True)
    # Deterministic scope enforcement (see annotate/scope.py) — enabled=False
    # (early_exit_on_ic_exclusion off) always yields excl_idx=None, so no
    # question can come out "skipped" unless the run opted in.
    parsed, _excl_idx = apply_scope_and_status(
        questions, parsed, enabled=cfg.early_exit_on_ic_exclusion, pass1_text=p1.text
    )

    if any(r.get("extraction_status") == "absent" for r in parsed):
        _dump_debug(batch_group_id, p1.text, p2.text, parsed, paper_id)

    # ---------- Build payloads ----------
    payloads = []
    for i, (question, result) in enumerate(zip(questions, parsed)):
        status = result.get("extraction_status", "ok")

        if status == "skipped":
            payloads.append(
                build_skipped_answer(
                    run_id=run.run_id, paper_id=paper_id, question=question,
                    extraction_detail=result.get("extraction_detail", "gated out"),
                )
            )
            continue

        if status == "absent":
            # Genuine omission (not gated out of scope) — a per-key error, NOT
            # a whole-group ExtractionError (the A5 parse-tolerance fix: one
            # missing key must not discard every other question's good answer).
            # Prefer the demotion's own extraction_detail (e.g. "pass-1 has no
            # answer block for ...") when set — it distinguishes a fabricated
            # Pass-2 answer from Pass-2 simply dropping the key.
            payloads.append(
                build_error_answer(
                    run_id=run.run_id, paper_id=paper_id, question=question,
                    extraction_detail=result.get(
                        "extraction_detail", f"key {question.key!r} not found in pass-2 output"
                    ),
                )
            )
            continue

        cited_error = result.get("cited_text_error")
        if cited_error:
            # Malformed cited_text means the whole item came back garbled — record
            # an extraction failure rather than a plausible-looking answer (see
            # annotate.parse.cited_text_violation).
            logger.error(
                "EXTRACTION FAILURE — run %d, paper %s, key %r: %s",
                run.run_id, paper_id, question.key, cited_error,
            )
            payloads.append(
                build_error_answer(
                    run_id=run.run_id, paper_id=paper_id, question=question,
                    extraction_detail=cited_error,
                )
            )
            continue

        verify = verify_citation(
            result.get("cited_text", ""),
            source_text,
            max_error_rate=citation_max_error_rate,
            max_ellipsis_gap=citation_max_ellipsis_gap,
        )
        cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]

        verify_list = verify_citations(
            result.get("cited_text", ""),
            source_text,
            max_error_rate=citation_max_error_rate,
            max_ellipsis_gap=citation_max_ellipsis_gap,
        )
        citations = build_citations(result.get("cited_text", ""), verify_list)

        if i == 0:
            raw_response = {
                "pass1_text": p1.text,
                "pass2_text": p2.text,
                "parse_result": result,
                "verify": verify,
                "text_source": cfg.text_source,
                "batch_group_id": batch_group_id,
                "p1_raw": p1.raw,
                "p2_raw": p2.raw,
            }
        else:
            raw_response = {
                "parse_result": result,
                "verify": verify,
                "text_source": cfg.text_source,
                "batch_group_id": batch_group_id,
            }

        # Usage/cost: attributed fully to the first answer in the group.
        # Reasoning model (p1) and format model (p2) are stored separately;
        # only p1 figures are sent to the server.
        if i == 0:
            tokens_total = p1.usage.total_tokens
            tokens_input = p1.usage.input_tokens
            tokens_output = p1.usage.output_tokens
            tokens_cached = p1.usage.cached_tokens
            tokens_reasoning = p1.usage.reasoning_tokens
            reasoning_content = p1.reasoning_content
            cost = p1.cost or Decimal(0)
            fmt_tokens_total = p2.usage.total_tokens
            fmt_tokens_input = p2.usage.input_tokens
            fmt_tokens_output = p2.usage.output_tokens
            fmt_tokens_cached = p2.usage.cached_tokens
            fmt_cost = p2.cost or Decimal(0)
            latency_ms = p1.latency_ms + p2.latency_ms
        else:
            tokens_total = tokens_input = tokens_output = tokens_cached = tokens_reasoning = 0
            reasoning_content = None
            cost = Decimal(0)
            fmt_tokens_total = fmt_tokens_input = fmt_tokens_output = fmt_tokens_cached = 0
            fmt_cost = Decimal(0)
            latency_ms = 0

        status_kwargs: dict = {}
        if status == "invalid":
            status_kwargs = {
                "extraction_status": "invalid",
                "extraction_detail": f"pass-2 reported unmappable value for {question.key!r}",
            }

        payload = build_llm_answer(
            run_id=run.run_id,
            paper_id=paper_id,
            question=question,
            value=result.get("value"),
            comment=result.get("comment", ""),
            cited_text=result.get("cited_text", ""),
            cited_text_verified=cited_text_verified,
            citations=citations,
            raw_response=raw_response,
            latency_ms=latency_ms,
            tokens_total=tokens_total,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cached=tokens_cached,
            tokens_reasoning=tokens_reasoning,
            reasoning_content=reasoning_content,
            cost=cost if cost else None,
            cost_currency=p1.cost_currency,
            fmt_tokens_total=fmt_tokens_total,
            fmt_tokens_input=fmt_tokens_input,
            fmt_tokens_output=fmt_tokens_output,
            fmt_tokens_cached=fmt_tokens_cached,
            fmt_cost=fmt_cost if fmt_cost else None,
            confidence=result.get("confidence"),
            **status_kwargs,
        )
        payloads.append(payload)

    return payloads


async def reformat_group(
    *,
    pass1_text: str,
    source_text: str,
    questions: list[Question],
    format_model: str,
    format_model_provider: str,
    format_structured_output: bool = True,
    format_temperature: float | None = None,
    format_model_params: dict | None = None,
    complete_fn: CompleteFn = llm_complete,
    citation_max_error_rate: float = 0.05,
    citation_max_ellipsis_gap: int = 300,
    request_timeout: float | None = None,
    require_status: bool = False,
) -> list[dict]:
    """Re-run only pass-2 (formatting) on existing pass-1 output.

    Returns one dict per question (in the same order as *questions*) with keys:
      parse_result, verify, cited_text_verified, citations, p2_text, p2_raw,
      fmt_tokens_{total,input,output,cached}, fmt_cost.
    Token/cost figures are attributed to index 0 only; the rest get zeros,
    matching the original attribution scheme in annotate_group().

    ``require_status`` is shared, opt-in-only: this function is reused
    UNCHANGED by arbitration's reformat entrypoint (see
    arbitrate_orchestrator.py), which never passes it, so the default False
    preserves that path's ``_RESPONSE_FORMAT`` / whole-group-failure behavior
    byte-for-byte. Reformat operates on rows whose scope was already decided
    at original-annotation time (it only re-runs Pass-2 formatting, not the
    IC-gate computation) — a missing key is still treated as a hard failure
    for the whole group either way; ``require_status=True`` only changes which
    signal that failure is read from (mechanical "absent" status vs. the
    legacy ``parse_error``), for parity with the main annotate path's schema.

    ``require_status=True`` also enforces the same demote-only fabrication
    guard as ``annotate/scope.py``: an entry claiming "ok"/"unmappable" whose
    key has no answer block in ``pass1_text`` (per
    ``parse.pass1_block_present``) is demoted to "absent" before the failure
    check below runs. Unlike
    ``apply_scope_and_status``, this function doesn't compute an IC exclusion
    point at all, so there's no exclusion-index-corruption risk to guard
    against here — but the same demoted entry now trips the existing
    ``status == "absent"`` failure path and raises ``ExtractionError`` for the
    whole group, rather than silently reformatting a fabricated answer. This
    check never runs when ``require_status=False`` (arbitration).
    """
    fmt_messages = build_format_messages(pass1_text, questions, require_status=require_status)

    p2_kwargs: dict = {
        "timeout": request_timeout or DEFAULT_REQUEST_TIMEOUT,
    }
    if format_temperature is not None:
        p2_kwargs["temperature"] = format_temperature
    if format_model_params:
        p2_kwargs.update(format_model_params)
    if format_structured_output:
        p2_kwargs["response_format"] = ANNOTATE_RESPONSE_FORMAT if require_status else _RESPONSE_FORMAT

    p2: LLMResult = await complete_fn(  # type: ignore[assignment]
        format_model,
        format_model_provider,
        fmt_messages,
        **p2_kwargs,
    )

    parsed = parse_structured_output(p2.text, [q.key for q in questions], annotate_mode=require_status)

    if require_status:
        for r in parsed:
            if r.get("status") in ("ok", "unmappable") and not pass1_block_present(pass1_text, r["key"]):
                logger.warning(
                    "reformat_group: Pass-2 claimed status=%r for key=%r but pass-1 has "
                    "no answer block for it — demoting to 'absent'",
                    r["status"], r["key"],
                )
                r["status"] = "absent"
                r["value"] = None
                r["cited_text"] = ""
                r["comment"] = ""
                r["confidence"] = None
                r["extraction_detail"] = f"pass-1 has no answer block for {r['key']!r}"
        failed = {
            r["key"]: r.get("extraction_detail", "not found in pass-2 output")
            for r in parsed if r.get("status") == "absent"
        }
    else:
        failed = {r["key"]: r["parse_error"] for r in parsed if "parse_error" in r}
    malformed = {
        r["key"]: r["cited_text_error"] for r in parsed if r.get("cited_text_error")
    }
    if malformed:
        failed = {**failed, **malformed}
    if failed:
        raise ExtractionError(failed)

    results = []
    for i, (question, result) in enumerate(zip(questions, parsed)):
        verify = verify_citation(
            result.get("cited_text", ""),
            source_text,
            max_error_rate=citation_max_error_rate,
            max_ellipsis_gap=citation_max_ellipsis_gap,
        )
        cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]

        verify_list = verify_citations(
            result.get("cited_text", ""),
            source_text,
            max_error_rate=citation_max_error_rate,
            max_ellipsis_gap=citation_max_ellipsis_gap,
        )
        citations = build_citations(result.get("cited_text", ""), verify_list)

        if i == 0:
            fmt_tokens_total = p2.usage.total_tokens
            fmt_tokens_input = p2.usage.input_tokens
            fmt_tokens_output = p2.usage.output_tokens
            fmt_tokens_cached = p2.usage.cached_tokens
            fmt_cost = p2.cost
        else:
            fmt_tokens_total = fmt_tokens_input = fmt_tokens_output = fmt_tokens_cached = 0
            fmt_cost = None
        results.append({
            "parse_result": result,
            "verify": verify,
            "cited_text_verified": cited_text_verified,
            "citations": citations,
            "p2_text": p2.text,
            "p2_raw": p2.raw,
            "fmt_tokens_total": fmt_tokens_total,
            "fmt_tokens_input": fmt_tokens_input,
            "fmt_tokens_output": fmt_tokens_output,
            "fmt_tokens_cached": fmt_tokens_cached,
            "fmt_cost": fmt_cost,
        })
    return results
