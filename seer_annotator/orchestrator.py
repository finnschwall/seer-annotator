"""Main annotation loop: run → paper → group → annotate → persist → post."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

_console = Console()

from .batching import resolve_groups
from .batch_runner import TRUNCATED_FINISH_REASONS, _execute_pass1, _execute_pass2, is_quota_exhausted
from .reply_quality import pass2_incomplete_reason
from .config import PipelineConfig, ProviderSettings, RunConfig, Settings, effective_run_config
from .progress import ProgressReporter, ProgressReporterProtocol
from .rate_limiter import PerProviderRateLimiter
from .seer_client import SeerClient
from .source_text import cached_full_text_for_run, full_text_for_run
from .store import Store
from .mapping import build_error_answer, build_llm_answer, build_skipped_answer, payload_is_error
from .annotate.citation import build_citations
from .annotate.parse import ANNOTATE_RESPONSE_FORMAT, ExtractionError, parse_structured_output
from .annotate.scope import apply_scope_and_status
from .annotate.verify import verify_citation, verify_citations

logger = logging.getLogger(__name__)


def _p1_drop_reason(
    cid: str,
    p1_errors: dict,
    p1_usage: dict,
    p1_error: "str | None",
) -> "tuple[str, dict]":
    """Why one cell has no usable Pass-1 output, and the evidence for it.

    Returns ``(detail, raw_response)`` for ``build_error_answer``. Three sources,
    most specific first:

    ``p1_errors[cid]``
        This cell's own failure — a batch item that errored, or an online call
        that raised. Carries the provider's own words via ``_format_llm_error``.
    ``p1_error``
        The phase aborted for everyone: the online probe call failed, so no cell
        has its own entry. Every cell nevertheless failed for this reason, and
        it used to be dropped here in favour of the generic sentence below —
        leaving the reader with "no output (API error/timeout); see log" for what
        was really an expired key or a rate limit, and the actual message only in
        the log.
    the generic sentence
        A cid that vanished for some other reason (e.g. loaded from a stale
        Pass-1 dump), where there is genuinely nothing more to say.

    The ``raw_response`` carries this cell's Pass-1 usage even though there is no
    Pass-1 *text* to go with it. That usage is the only record of ``max_tokens``
    and ``finish_reason``, which is what lets a reader downstream tell a
    truncated Pass 1 apart from a provider error — and a truncated Pass 1 has
    already had its text popped by ``_demote_truncated_p1``.
    """
    detail = p1_errors.get(cid) or p1_error or (
        "pass1 failed — no output (API error/timeout); see log"
    )
    usage = p1_usage.get(cid) or {}
    code = (
        "pass1_truncated"
        if usage.get("finish_reason") in TRUNCATED_FINISH_REASONS
        else "pass1_provider_error"
    )
    raw_response = {
        "p1_usage": usage,
        "batch_group_id": cid,
        "diagnostics": [{"phase": "pass1", "code": code, "detail": detail}],
    }
    return detail, raw_response


def _chunks(lst: list, size: int) -> list[list]:
    """Split lst into sublists of at most size elements. size<=0 → one chunk."""
    if size <= 0:
        return [lst] if lst else []
    return [lst[i : i + size] for i in range(0, len(lst), size)] if lst else []


def _parse_save_post_tail(
    *,
    p1_texts: dict,
    p1_usage: dict,
    p2_texts: dict,
    p2_usage: dict,
    pending_cells: dict,
    source_texts: dict,
    run,
    cfg,
    store: Store,
    fail_fast: bool,
    on_payload_saved: "Callable[[dict], None] | None" = None,
    excluded_papers: "dict[int, str] | None" = None,
) -> "ExtractionError | None":
    """Parse/verify/save answers from p2_texts. Returns first ExtractionError if any.

    ``on_payload_saved``, when given, is called once per built payload right after
    it's persisted — used by the progress heartbeat to track cells_done/cells_error
    without a separate store query.

    ``excluded_papers``, when given, is mutated in place: whenever a group's
    scope computation finds an IC exclusion, ``{paper_id: detail}`` is recorded
    — used by the round-based per_question/size early-exit loop (A7, see
    ``_run_chunk_with_early_exit``) to carry the exclusion decision forward to
    later rounds/groups for the same paper without re-deriving it.
    """
    first_extraction_error: ExtractionError | None = None

    for cid in p2_texts:
        if cid not in pending_cells:
            continue
        paper, group, group_idx = pending_cells[cid]
        p1_text = p1_texts.get(cid, "")
        p2_text = p2_texts[cid]

        u = p1_usage.get(cid, {})
        tok_input     = u.get("input_tokens", 0)
        tok_output    = u.get("output_tokens", 0)
        tok_cached    = u.get("cache_read_tokens", 0)
        tok_reasoning = u.get("reasoning_tokens", 0) or 0
        p1_reasoning_content = u.get("reasoning_content")
        p1_cost    = u.get("cost")
        p1_latency = u.get("latency_ms", 0) or 0

        fu = p2_usage.get(cid, {})
        fmt_input  = fu.get("input_tokens", 0)
        fmt_output = fu.get("output_tokens", 0)
        fmt_cached = fu.get("cache_read_tokens", 0)
        fmt_total  = fmt_input + fmt_output + fmt_cached
        fmt_cost   = fu.get("cost")
        p2_latency = fu.get("latency_ms", 0) or 0

        def _save_extraction_error(question, detail, *, code, label, extra=None):
            """Persist a per-question error answer and log it.

            One cell's bad reply costs that cell, not the run — this is the shared
            tail for every "we cannot trust this answer" exit below.
            """
            logger.error(
                "%s — run %d, paper %d, group %d, key %r: %s",
                label, run.run_id, paper.paper_id, group_idx, question.key, detail,
            )
            diagnostic = {"phase": "pass2", "code": code, "detail": detail}
            if extra:
                diagnostic.update(extra)
            error_payload = build_error_answer(
                run_id=run.run_id, paper_id=paper.paper_id, question=question,
                extraction_detail=detail,
                raw_response={
                    "pass1_text": p1_text,
                    "pass2_text": p2_text,
                    "p1_usage": p1_usage.get(cid, {}),
                    "p2_usage": p2_usage.get(cid, {}),
                    "diagnostics": [diagnostic],
                    "batch_group_id": cid,
                },
            )
            store.save_answer(run.run_id, paper.paper_id, question.version_id, error_payload, cid)
            if on_payload_saved is not None:
                on_payload_saved(error_payload)

        # Did this reply finish? A Pass 2 that looped or ran out of room still
        # parses — json_repair closes the braces it never wrote — and the items
        # that got out are whichever ones came first in the schema's field
        # order, not the ones that were right. Fail the whole group on the reply
        # rather than let the parser read values off salvage.
        incomplete = pass2_incomplete_reason(p2_text, fu)
        if incomplete is not None:
            code, detail = incomplete
            for question in group:
                _save_extraction_error(
                    question, detail, code=code, label="EXTRACTION FAILURE",
                )
            if fail_fast and first_extraction_error is None:
                first_extraction_error = ExtractionError({q.key: detail for q in group})
            continue

        try:
            parsed = parse_structured_output(
                p2_text, [q.key for q in group], annotate_mode=True, require_status=True,
            )
            # Deterministic scope enforcement — authoritative over the model (see
            # annotate/scope.py). enabled=False (early_exit_on_ic_exclusion off)
            # short-circuits to excl_idx=None every time, so no cell can ever come
            # out "skipped" unless the run opted in.
            parsed, excl_idx = apply_scope_and_status(
                group, parsed, enabled=cfg.early_exit_on_ic_exclusion,
                # `or None`: p1_text defaults to "" above, and an empty string would
                # read as "pass-1 answered nothing" and demote the whole group.
                pass1_text=p1_text or None,
            )
        except Exception as exc:
            # Nothing above this tail catches — run_pipeline calls it unguarded and
            # asyncio.run re-raises — so an escape here ends the job and leaves every
            # remaining paper unattempted. Fail the cell instead.
            detail = f"pass-2 parse failed: {type(exc).__name__}: {exc}"
            logger.exception(
                "PARSE CRASH — run %d, paper %d, group %d", run.run_id, paper.paper_id, group_idx,
            )
            for question in group:
                _save_extraction_error(
                    question, detail, code="pass2_parse_crash", label="EXTRACTION FAILURE",
                )
            if fail_fast and first_extraction_error is None:
                first_extraction_error = ExtractionError({q.key: detail for q in group})
            continue

        if excl_idx is not None and excluded_papers is not None:
            excluded_papers[paper.paper_id] = f"gated out: {group[excl_idx].key}=false"

        for i, (question, result) in enumerate(zip(group, parsed)):
            try:
                status = result.get("extraction_status", "ok")

                if status == "skipped":
                    payload = build_skipped_answer(
                        run_id=run.run_id, paper_id=paper.paper_id, question=question,
                        extraction_detail=result.get("extraction_detail", "gated out"),
                    )
                    store.save_answer(run.run_id, paper.paper_id, question.version_id, payload, cid)
                    if on_payload_saved is not None:
                        on_payload_saved(payload)
                    continue

                if status == "absent":
                    # Genuine omission (no exclusion moots this key) — per-key error,
                    # NOT a whole-group failure (the A5 parse-tolerance fix). Prefer
                    # the demotion's own extraction_detail (e.g. "pass-1 has no
                    # answer block for ...") over the generic message — it tells
                    # apart Pass-1 never answering from Pass-2 dropping the key.
                    err_detail = result.get(
                        "extraction_detail", f"key {question.key!r} not found in pass-2 output"
                    )
                    _save_extraction_error(
                        question, err_detail, code="pass2_parse_missing_keys",
                        label="EXTRACTION FAILURE", extra={"missing_keys": [question.key]},
                    )
                    if fail_fast and first_extraction_error is None:
                        first_extraction_error = ExtractionError({question.key: err_detail})
                    continue

                cited_error = result.get("cited_text_error")
                if cited_error:
                    # The item's JSON did not survive parsing: json_repair
                    # resynchronised the malformed part into a nested list inside
                    # cited_text, swallowing this item's comment/confidence/status
                    # along with it. The value we would save is not what the model
                    # wrote, so report an extraction failure instead of an answer
                    # that looks normal. See annotate.parse.cited_text_violation.
                    _save_extraction_error(
                        question, cited_error, code="pass2_malformed_cited_text",
                        label="EXTRACTION FAILURE",
                    )
                    if fail_fast and first_extraction_error is None:
                        first_extraction_error = ExtractionError({question.key: cited_error})
                    continue

                item_error = result.get("item_error")
                if item_error:
                    # The reply stopped partway through this item, and the fields
                    # it never reached fell back to their defaults — including
                    # status="ok". The reply gate above catches this whenever the
                    # reply itself shows the damage; this catches the case where
                    # it does not (see annotate.parse._incomplete_item_violation).
                    _save_extraction_error(
                        question, item_error, code="pass2_incomplete_item",
                        label="EXTRACTION FAILURE",
                    )
                    if fail_fast and first_extraction_error is None:
                        first_extraction_error = ExtractionError({question.key: item_error})
                    continue

                verify = verify_citation(
                    result.get("cited_text", ""),
                    source_texts.get(paper.paper_id, ""),
                    max_error_rate=cfg.citation_max_error_rate,
                    max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                )
                cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]

                verify_list = verify_citations(
                    result.get("cited_text", ""),
                    source_texts.get(paper.paper_id, ""),
                    max_error_rate=cfg.citation_max_error_rate,
                    max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                )
                citations = build_citations(result.get("cited_text", ""), verify_list)

                raw_response = {
                    "pass1_text": p1_text,
                    "pass2_text": p2_text,
                    "parse_result": result,
                    "verify": verify,
                    "text_source": cfg.text_source,
                    "batch_group_id": cid,
                    "batch_mode": True,
                    "p1_usage": u,
                    "p2_usage": p2_usage.get(cid, {}),
                }
                # status == "invalid" is Pass-2's own mechanical determination
                # ("content present but unmappable") — pass it through explicitly
                # rather than letting build_llm_answer re-derive extraction_status
                # from _map_value, since that would fall back to "ok" for a value
                # that happens to coincidentally validate.
                status_kwargs: dict = {}
                if status == "invalid":
                    status_kwargs = {
                        "extraction_status": "invalid",
                        "extraction_detail": f"pass-2 reported unmappable value for {question.key!r}",
                    }
                payload = build_llm_answer(
                    run_id=run.run_id,
                    paper_id=paper.paper_id,
                    question=question,
                    value=result.get("value"),
                    comment=result.get("comment", ""),
                    cited_text=result.get("cited_text", ""),
                    cited_text_verified=cited_text_verified,
                    citations=citations,
                    raw_response=raw_response,
                    latency_ms=(p1_latency + p2_latency) if i == 0 else None,
                    tokens_total=tok_input + tok_output + tok_cached if i == 0 else None,
                    tokens_input=tok_input if i == 0 else None,
                    tokens_output=tok_output if i == 0 else None,
                    tokens_cached=tok_cached if i == 0 else None,
                    tokens_reasoning=tok_reasoning if i == 0 else None,
                    reasoning_content=p1_reasoning_content if i == 0 else None,
                    cost=p1_cost if i == 0 else None,
                    cost_currency="USD",
                    fmt_tokens_total=fmt_total if i == 0 else None,
                    fmt_tokens_input=fmt_input if i == 0 else None,
                    fmt_tokens_output=fmt_output if i == 0 else None,
                    fmt_tokens_cached=fmt_cached if i == 0 else None,
                    fmt_cost=fmt_cost if i == 0 else None,
                    confidence=result.get("confidence"),
                    **status_kwargs,
                )
                store.save_answer(run.run_id, paper.paper_id, question.version_id, payload, cid)
                if on_payload_saved is not None:
                    on_payload_saved(payload)
            except Exception as exc:
                # Same containment as the parse guard above, one level in: verification,
                # payload building or the store raising for one question must not take
                # the rest of the batch (or the run) with it.
                detail = f"pass-2 post-processing failed: {type(exc).__name__}: {exc}"
                logger.exception(
                    "POST-PROCESSING CRASH — run %d, paper %d, group %d, key %r",
                    run.run_id, paper.paper_id, group_idx, question.key,
                )
                try:
                    _save_extraction_error(
                        question, detail, code="pass2_unexpected_error",
                        label="EXTRACTION FAILURE",
                    )
                except Exception:
                    logger.exception(
                        "Could not even save the error answer for run %d, paper %d, key %r",
                        run.run_id, paper.paper_id, question.key,
                    )
                if fail_fast and first_extraction_error is None:
                    first_extraction_error = ExtractionError({question.key: detail})
                continue

    return first_extraction_error


async def _run_rounds_with_early_exit(
    *,
    run,
    cfg,
    chunk_papers: list,
    source_texts: dict,
    groups_def: list,
    store: Store,
    settings,
    dry_run: bool,
    chunk_i: int,
    sem,
    limiter,
    on_cell_done,
    on_total_known,
    on_p2_start,
    on_p2_advance,
    on_payload_saved,
) -> tuple[int, bool, list[ExtractionError]]:
    """True early-exit for per_question/size batching (A7/A8) — one ROUND per
    group in ``groups_def``, instead of building every paper's every group up
    front (which would issue every LLM call regardless of exclusion — the
    behavior this replaces, and the only reason it's a separate code path
    from the normal single-shot chunk processing below).

    Round r processes ``groups_def[r]`` for every paper still "active": has a
    resolved ``source_texts`` entry, and hasn't already been marked excluded
    by an earlier round. The moment a paper's group resolves to an IC
    exclusion (via ``apply_scope_and_status`` inside ``_parse_save_post_tail``,
    threaded through via the ``excluded_papers`` out-param), it drops out of
    all LATER rounds — no further LLM calls are issued for it, and each later
    round emits a ``skipped`` LLMAnswer for that paper's questions directly
    (A6/A8: these count as done/terminal cells, same as any other payload,
    via ``on_payload_saved``).

    Not used for ``"all"`` batching (``len(groups_def) == 1``, one atomic
    group — there's no "later round" to skip; its savings come from truncated
    Pass-1/Pass-2 *output* once the model obeys the stop instruction, see
    prompt.py, plus this module's own per-key relabeling either way).

    Returns ``(chunk_failed_count, run_had_fatal_error, extraction_errors)``,
    mirroring the accounting the non-early-exit path keeps inline in
    ``run_pipeline`` (``run_cell_errors`` / ``run_had_fatal_error`` /
    ``first_error``).
    """
    excluded_papers: dict[int, str] = {}
    chunk_failed = 0
    run_had_fatal_error = False
    extraction_errors: list[ExtractionError] = []

    eligible_paper_ids = {p.paper_id for p in chunk_papers if p.paper_id in source_texts}

    for round_idx, group in enumerate(groups_def):
        # Papers already excluded by an earlier round: emit `skipped` rows for
        # this round's questions directly — no LLM call, this IS the saving.
        for paper in chunk_papers:
            if paper.paper_id not in eligible_paper_ids:
                continue
            detail = excluded_papers.get(paper.paper_id)
            if detail is None:
                continue
            for q in group:
                skip_payload = build_skipped_answer(
                    run_id=run.run_id, paper_id=paper.paper_id, question=q,
                    extraction_detail=detail,
                )
                store.save_answer(run.run_id, paper.paper_id, q.version_id, skip_payload)
                if on_payload_saved is not None:
                    on_payload_saved(skip_payload)

        round_papers = [
            p for p in chunk_papers
            if p.paper_id in eligible_paper_ids and p.paper_id not in excluded_papers
        ]
        if not round_papers:
            continue

        # A distinct kv/dump-file scope per round (see _execute_pass1_with_groups's
        # chunk_i docstring for why this matters) — a plain string is fine, it's
        # only ever used inside f-strings/filenames, never compared as an int.
        round_key = f"{chunk_i}-r{round_idx}"

        round_pending: dict[str, tuple] = {}
        p1_texts, p1_usage, p1_errors, p1_error = await _execute_pass1(
            run, cfg, round_papers, source_texts, round_pending,
            store, settings, dry_run, groups_def=[group],
            chunk_i=round_key, sem=sem, limiter=limiter,
            on_cell_done=on_cell_done, on_total_known=on_total_known,
        )

        chunk_failed += len(round_pending) - len(p1_texts)

        if p1_error:
            logger.error("P1 aborted (round %d) — %s", round_idx, p1_error)
            run_had_fatal_error = True
            # No `break` yet: the abort still has to be recorded on the cells it
            # cost, or this round's papers end up with no answer and no error —
            # indistinguishable from never having been in scope. The drop loop
            # below writes them all (a fatal P1 returns no texts, so every cell
            # in `round_pending` is a drop), and the break follows it.

        for cid, (paper, grp, group_idx) in round_pending.items():
            if cid in p1_texts:
                continue
            cell_error_detail, cell_raw = _p1_drop_reason(cid, p1_errors, p1_usage, p1_error)
            for q in grp:
                p1_drop_payload = build_error_answer(
                    run_id=run.run_id, paper_id=paper.paper_id, question=q,
                    extraction_detail=cell_error_detail,
                    raw_response=cell_raw,
                )
                store.save_answer(run.run_id, paper.paper_id, q.version_id, p1_drop_payload)
                if on_payload_saved is not None:
                    on_payload_saved(p1_drop_payload)
                if cid in p1_errors or p1_error:
                    store.mark_failed(run.run_id, paper.paper_id, q.version_id, cell_error_detail)

        if p1_error:
            break  # mirrors the non-round path: a fatal P1 error aborts the run

        round_pending_p1 = {cid: t for cid, t in p1_texts.items() if cid in round_pending}
        p2_texts, p2_usage, p2_errors = await _execute_pass2(
            run, cfg, round_pending_p1, round_pending,
            store, settings, dry_run,
            chunk_i=round_key, sem=sem, limiter=limiter,
            on_p2_start=on_p2_start, on_p2_advance=on_p2_advance,
            response_format=ANNOTATE_RESPONSE_FORMAT, require_status=True,
        )

        for cid, err_detail in p2_errors.items():
            if cid not in round_pending:
                continue
            paper, grp, group_idx = round_pending[cid]
            for q in grp:
                p2_drop_payload = build_error_answer(
                    run_id=run.run_id, paper_id=paper.paper_id, question=q,
                    extraction_detail=err_detail,
                    # Same shape as the non-round path's P2-drop payload: without
                    # the explicit code, a consumer has to guess the failing pass
                    # from the wording of `err_detail`.
                    raw_response={
                        "pass1_text": p1_texts.get(cid, ""),
                        "p1_usage": p1_usage.get(cid, {}),
                        "p2_usage": p2_usage.get(cid, {}),
                        "batch_group_id": cid,
                        "diagnostics": [{
                            "phase": "pass2", "code": "pass2_provider_error",
                            "detail": err_detail,
                        }],
                    },
                )
                store.save_answer(run.run_id, paper.paper_id, q.version_id, p2_drop_payload)
                if on_payload_saved is not None:
                    on_payload_saved(p2_drop_payload)
                store.mark_failed(run.run_id, paper.paper_id, q.version_id, err_detail)

        err = _parse_save_post_tail(
            p1_texts=p1_texts,
            p1_usage=p1_usage,
            p2_texts=p2_texts,
            p2_usage=p2_usage,
            pending_cells=round_pending,
            source_texts=source_texts,
            run=run,
            cfg=cfg,
            store=store,
            fail_fast=cfg.fail_fast,
            on_payload_saved=on_payload_saved,
            excluded_papers=excluded_papers,
        )
        if err is not None:
            extraction_errors.append(err)

    return chunk_failed, run_had_fatal_error, extraction_errors


async def run_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
    store: Store | None = None,
    client: SeerClient | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
    progress_url: str | None = None,
    reporter_factory: "Callable[[int], ProgressReporterProtocol] | None" = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> None:
    from .seer_client import DryRunSeerClient

    # Pipeline-wide settings: code defaults < run_defaults < CLI overrides
    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)
    if concurrency is not None:
        pipeline_cfg.concurrency = concurrency
    if rpm is not None:
        pipeline_cfg.per_provider_rpm = rpm

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params
    _litellm.suppress_debug_info = True

    store = store or Store(settings.runtime.store_path)
    client = client or (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    sem = asyncio.Semaphore(pipeline_cfg.concurrency)
    limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

    all_run_errors: list[str] = []
    any_run_failed = False

    with progress:
        run_task = progress.add_task("Runs", total=len(runs))
        chunk_task = progress.add_task("  Chunks", total=0, visible=False)
        cell_task = progress.add_task("    Cells", total=0, visible=False)
        p2_task = progress.add_task("    Pass 2", total=0, visible=False)

        for run_idx, run in enumerate(runs):
            progress.update(
                run_task,
                description=f"Run {run.run_id} [cyan]({run.name})[/cyan]",
            )
            logger.debug("Run %d (%s) starting", run.run_id, run.name)

            cfg = effective_run_config(run.config, settings.run_defaults)
            _litellm.drop_params = cfg.drop_params
            groups_def = resolve_groups(cfg, pipeline.questions)

            chunks = _chunks(papers, cfg.chunk_papers)
            progress.reset(chunk_task, total=len(chunks), visible=True)

            first_error: list[ExtractionError] = []
            run_cell_errors = 0
            run_had_fatal_error = False
            # Set once any cell's error text says the account is out of
            # credits/quota (see is_quota_exhausted). Unlike an ordinary
            # per-cell failure, this always stops the run -- continuing to
            # burn through further chunks against an empty account cannot
            # succeed, so it overrides cfg.fail_fast rather than obeying it.
            quota_exhausted = False

            reporter = (
                reporter_factory(run.run_id)
                if reporter_factory is not None
                else ProgressReporter(progress_url, pipeline.api_token, run.run_id)
            )
            cells_total = len(papers) * len(pipeline.questions)
            # {cell key: is_error} for every cell counted so far, seeded from the store
            # rather than starting empty: a run using the batch API parks on
            # BatchPendingError and re-enters run_pipeline from the top, and the cells an
            # earlier invocation finished are dropped before they can be counted. Starting
            # at 0 made a resumed multi-chunk run report its last chunk only (e.g. 36/316),
            # which SEER correctly refuses to accept as a success — so a finished job was
            # marked failed. Keyed per cell, not a bare total, because a resume also
            # re-processes every cell that is not done/posted; each must count once
            # however many times it is seen. See Store.finished_cells.
            counted_cells = store.finished_cells(
                run.run_id,
                [p.paper_id for p in papers],
                [q.version_id for q in pipeline.questions],
            )
            cell_counters = {
                "done": len(counted_cells),
                "error": sum(1 for is_error in counted_cells.values() if is_error),
            }
            if counted_cells:
                logger.info(
                    "Run %d: resuming with %d/%d cells already done (%d error)",
                    run.run_id, cell_counters["done"], cells_total, cell_counters["error"],
                )

            # Tracks progress within whichever pass is currently in flight for the
            # active chunk ("pass1" = extraction, "pass2" = structured-output
            # formatting) — reset per chunk and again at the pass1->pass2 handoff.
            # Distinct from cell_counters, which only advances once both passes
            # finish for a cell; this is what lets the heartbeat convey sub-chunk
            # progress instead of sitting frozen until the whole chunk completes.
            phase_state = {"phase": None, "done": 0, "total": 0}

            # Multi-run pipelines (a whole ExperimentSetup run sequentially in one
            # invocation) share this same --progress-url across every run, so the
            # message is the only signal that tells the poller which run is active.
            def _progress_message(suffix: str, _idx=run_idx, _run=run) -> str:
                if len(runs) <= 1:
                    return suffix
                return f"run {_idx + 1}/{len(runs)} ({_run.name}): {suffix}"

            # Fire-and-forget per-cell heartbeats, throttled — supplements the
            # guaranteed start/chunk-boundary/end heartbeats below with something
            # closer to real-time progress for runs that fit in a single chunk
            # (chunk-boundary heartbeats alone would otherwise sit at 0 done for
            # the whole run and then jump straight to 100%).
            _heartbeat_tasks: set = set()
            _last_heartbeat_ts = [0.0]

            def _maybe_heartbeat(min_interval: float = 1.5) -> None:
                now = time.monotonic()
                if now - _last_heartbeat_ts[0] < min_interval:
                    return
                _last_heartbeat_ts[0] = now
                task = asyncio.create_task(reporter.heartbeat(
                    status="running", cells_total=cells_total,
                    cells_done=cell_counters["done"], cells_error=cell_counters["error"],
                    message=_progress_message(f"{cell_counters['done']}/{cells_total} cells done"),
                    chunk_index=chunk_i, chunk_total=len(chunks),
                    phase=phase_state["phase"], phase_done=phase_state["done"],
                    phase_total=phase_state["total"],
                ))
                _heartbeat_tasks.add(task)
                task.add_done_callback(_heartbeat_tasks.discard)

            def _count_payload(payload: dict, _c=cell_counters, _seen=counted_cells) -> None:
                key = (payload.get("paper"), payload.get("question_version"))
                if key[0] is None or key[1] is None:
                    key = object()  # unidentifiable payload: count it, never collapse it
                is_error = payload_is_error(payload)
                if key in _seen:
                    # Re-processed cell (its group had a failure): correct its verdict
                    # in place instead of counting it a second time.
                    _c["error"] -= 1 if _seen[key] else 0
                else:
                    _c["done"] += 1
                _c["error"] += 1 if is_error else 0
                _seen[key] = is_error
                _maybe_heartbeat()

            await reporter.heartbeat(
                status="running", cells_total=cells_total,
                cells_done=cell_counters["done"], cells_error=cell_counters["error"],
                message=_progress_message("starting"),
            )

            for chunk_i, chunk_papers in enumerate(chunks):
                if should_cancel is not None and should_cancel():
                    # Cooperative cancel: stop processing further chunks/runs. Drain
                    # any in-flight per-cell heartbeats first so a late one can't
                    # land after (and overwrite) this terminal "canceled" heartbeat.
                    if _heartbeat_tasks:
                        await asyncio.gather(*_heartbeat_tasks, return_exceptions=True)
                    await reporter.heartbeat(
                        status="canceled",
                        cells_total=cells_total,
                        cells_done=cell_counters["done"],
                        cells_error=cell_counters["error"],
                        message=_progress_message("canceled"),
                    )
                    return
                progress.update(chunk_task, description=f"  Chunk {chunk_i + 1}/{len(chunks)}")
                progress.reset(cell_task, total=None, visible=False)
                phase_state["phase"] = None
                phase_state["done"] = 0
                phase_state["total"] = 0

                def _on_total(n, _t=cell_task):
                    progress.update(_t, total=n, visible=n > 0)
                    phase_state["phase"] = "pass1"
                    phase_state["done"] = 0
                    phase_state["total"] = n

                def _on_cell_done(_t=cell_task):
                    progress.advance(_t)
                    phase_state["done"] += 1
                    _maybe_heartbeat()

                # Resolve source texts for this chunk
                source_texts: dict[int, str] = {}
                no_ocr_papers: list = []

                for paper in chunk_papers:
                    if cfg.text_source == "full_text":
                        source = await full_text_for_run(
                            client, store, paper.paper_id, run.run_id,
                        )
                        if source is None:
                            logger.warning("No OCR for paper %d — posting error for all questions", paper.paper_id)
                            no_ocr_papers.append(paper)
                            for q in pipeline.questions:
                                no_ocr_payload = build_error_answer(run_id=run.run_id, paper_id=paper.paper_id, question=q, extraction_detail="no_ocr")
                                store.save_answer(run.run_id, paper.paper_id, q.version_id, no_ocr_payload)
                                _count_payload(no_ocr_payload)
                        else:
                            source_texts[paper.paper_id] = source
                    else:
                        source_texts[paper.paper_id] = paper.abstract

                def _on_p2_start(n: int, desc: str, _t=p2_task) -> None:
                    progress.reset(_t, total=n, visible=n > 0)
                    progress.update(_t, description=f"    {desc}")
                    phase_state["phase"] = "pass2"
                    phase_state["done"] = 0
                    phase_state["total"] = n

                def _on_p2_advance(_t=p2_task) -> None:
                    progress.advance(_t)
                    phase_state["done"] += 1
                    _maybe_heartbeat()

                # A7: per_question/size batching with early_exit_on_ic_exclusion
                # gets a true round-based early exit — groups_def has >1 group,
                # and the round loop skips issuing LLM calls entirely for any
                # paper already excluded by an earlier round (see
                # _run_rounds_with_early_exit). "all" batching (one atomic
                # group, len(groups_def) == 1) and/or the flag being off both
                # fall through to the original single-shot path below,
                # unchanged.
                if cfg.early_exit_on_ic_exclusion and len(groups_def) > 1:
                    chunk_failed, fatal, round_errors = await _run_rounds_with_early_exit(
                        run=run, cfg=cfg, chunk_papers=chunk_papers,
                        source_texts=source_texts, groups_def=groups_def,
                        store=store, settings=settings, dry_run=dry_run,
                        chunk_i=chunk_i, sem=sem, limiter=limiter,
                        on_cell_done=_on_cell_done, on_total_known=_on_total,
                        on_p2_start=_on_p2_start, on_p2_advance=_on_p2_advance,
                        on_payload_saved=_count_payload,
                    )
                    run_cell_errors += chunk_failed
                    if fatal:
                        msg = f"Run {run.run_id} ({run.name}): pass-1 aborted during early-exit round"
                        all_run_errors.append(msg)
                        run_had_fatal_error = True
                    if not quota_exhausted and any(is_quota_exhausted(str(e)) for e in round_errors):
                        logger.error("Run %d: provider out of credits/quota (early-exit round) — stopping this run", run.run_id)
                        run_had_fatal_error = True
                        quota_exhausted = True
                    first_error.extend(round_errors)
                else:
                    # Phase 1
                    pending_cells: dict[str, tuple] = {}
                    p1_texts, p1_usage, p1_errors, p1_error = await _execute_pass1(
                        run, cfg, chunk_papers, source_texts, pending_cells,
                        store, settings, dry_run, groups_def=groups_def,
                        chunk_i=chunk_i, sem=sem, limiter=limiter,
                        on_cell_done=_on_cell_done, on_total_known=_on_total,
                    )

                    n_pending = len(pending_cells)
                    n_got_p1 = len(p1_texts)
                    chunk_failed = n_pending - n_got_p1
                    run_cell_errors += chunk_failed

                    if p1_error:
                        msg = f"Run {run.run_id} ({run.name}): {p1_error}"
                        all_run_errors.append(msg)
                        logger.error("P1 aborted — %s", msg)
                        # p1_error signals a fatal, run-aborting failure (e.g. bad model name/API
                        # key — the "probe first call" path in _execute_pass1_with_groups), not a
                        # per-cell parsing issue. Treat it like first_error for fail_fast purposes
                        # so a totally broken run is actually reported as failed (and stops a
                        # multi-run "Run all" pipeline) instead of silently completing 0/N cells
                        # and reporting "succeeded".
                        run_had_fatal_error = True
                        if is_quota_exhausted(p1_error):
                            quota_exhausted = True
                    elif chunk_failed > 0:
                        logger.warning("Run %d: %d/%d cells failed in chunk %d", run.run_id, chunk_failed, n_pending, chunk_i + 1)

                    if not quota_exhausted and any(is_quota_exhausted(e) for e in p1_errors.values()):
                        # The probe call succeeded but a later concurrent P1 call hit the
                        # same wall (only the first call is probed before the rest are
                        # fired together — see _execute_pass1). Those already-in-flight
                        # calls can't be un-sent, but nothing later in this run should
                        # keep trying against an empty account.
                        logger.error("Run %d: provider out of credits/quota (pass 1) — stopping this run", run.run_id)
                        run_had_fatal_error = True
                        quota_exhausted = True

                    # Cells that entered pending_cells but got no Pass-1 output at all
                    # (API error/timeout inside _execute_pass1) would otherwise vanish
                    # silently: Phase 2 only iterates pending_p1 (cids WITH p1 text) and
                    # the parse/save tail only iterates p2_texts, so a dropped cid is never
                    # touched again. Post an explicit error answer for each such cell now,
                    # mirroring the no-OCR error path above, so the failure is visible
                    # instead of the paper just quietly missing an answer.
                    for cid, (paper, group, group_idx) in pending_cells.items():
                        if cid in p1_texts:
                            continue
                        # See _p1_drop_reason for which of the three possible
                        # reasons this cell gets, and why its Pass-1 usage rides
                        # along without any Pass-1 text.
                        cell_error_detail, cell_raw = _p1_drop_reason(
                            cid, p1_errors, p1_usage, p1_error,
                        )
                        for q in group:
                            p1_drop_payload = build_error_answer(
                                run_id=run.run_id, paper_id=paper.paper_id, question=q,
                                extraction_detail=cell_error_detail,
                                raw_response=cell_raw,
                            )
                            store.save_answer(run.run_id, paper.paper_id, q.version_id, p1_drop_payload)
                            _count_payload(p1_drop_payload)
                            if cid in p1_errors or p1_error:
                                # Flip the store's local status away from 'done' (what
                                # save_answer always sets) to 'failed' so should_skip_cell
                                # treats this as retryable on the next run, instead of
                                # permanently poisoning it the way a deterministic parse
                                # error is (those legitimately stay 'done').
                                # The error answer saved just above is still posted by
                                # the per-paper loop below: what gets posted is decided
                                # by store.posted_at, not by this status.
                                store.mark_failed(run.run_id, paper.paper_id, q.version_id, cell_error_detail)

                    # Phase 2
                    pending_p1 = {cid: t for cid, t in p1_texts.items() if cid in pending_cells}

                    p2_texts, p2_usage, p2_errors = await _execute_pass2(
                        run, cfg, pending_p1, pending_cells,
                        store, settings, dry_run,
                        chunk_i=chunk_i, sem=sem, limiter=limiter,
                        on_p2_start=_on_p2_start,
                        on_p2_advance=_on_p2_advance,
                        response_format=ANNOTATE_RESPONSE_FORMAT,
                        require_status=True,
                    )

                    # Cells that had a Pass-1 result but failed Pass-2 (batch item
                    # errored/canceled/expired, or an online P2 call that raised) would
                    # otherwise vanish the same way a dropped Pass-1 cell would — the
                    # parse/save tail below only iterates p2_texts. Mirror the p1-drop
                    # handling above so these are posted as visible error answers too.
                    for cid, err_detail in p2_errors.items():
                        if cid not in pending_cells:
                            continue
                        paper, group, group_idx = pending_cells[cid]
                        for q in group:
                            p2_drop_payload = build_error_answer(
                                run_id=run.run_id, paper_id=paper.paper_id, question=q,
                                extraction_detail=err_detail,
                                raw_response={
                                    "pass1_text": p1_texts.get(cid, ""),
                                    "p1_usage": p1_usage.get(cid, {}),
                                    "p2_usage": p2_usage.get(cid, {}),
                                    "batch_group_id": cid,
                                    "diagnostics": [{"phase": "pass2", "code": "pass2_provider_error", "detail": err_detail}],
                                },
                            )
                            store.save_answer(run.run_id, paper.paper_id, q.version_id, p2_drop_payload)
                            _count_payload(p2_drop_payload)
                            store.mark_failed(run.run_id, paper.paper_id, q.version_id, err_detail)

                    if not quota_exhausted and any(is_quota_exhausted(e) for e in p2_errors.values()):
                        logger.error("Run %d: provider out of credits/quota (pass 2) — stopping this run", run.run_id)
                        run_had_fatal_error = True
                        quota_exhausted = True

                    # Parse / verify / save
                    err = _parse_save_post_tail(
                        p1_texts=p1_texts,
                        p1_usage=p1_usage,
                        p2_texts=p2_texts,
                        p2_usage=p2_usage,
                        pending_cells=pending_cells,
                        source_texts=source_texts,
                        run=run,
                        cfg=cfg,
                        store=store,
                        fail_fast=cfg.fail_fast,
                        on_payload_saved=_count_payload,
                    )
                    if err is not None:
                        first_error.append(err)

                # Post per paper in chunk (includes no-OCR error records)
                for paper in chunk_papers:
                    unposted = store.get_unposted(run.run_id, paper.paper_id)
                    if unposted:
                        try:
                            await client.post_answers_bulk(unposted)
                            if not dry_run:
                                version_ids = [p["question_version"] for p in unposted]
                                store.mark_posted(run.run_id, paper.paper_id, version_ids)
                            logger.info("Posted %d answers for paper %d", len(unposted), paper.paper_id)
                        except Exception as exc:
                            logger.error("Post failed for paper %d: %s", paper.paper_id, exc)

                progress.advance(chunk_task)

                await reporter.heartbeat(
                    status="running",
                    cells_total=cells_total,
                    cells_done=cell_counters["done"],
                    cells_error=cell_counters["error"],
                    message=_progress_message(f"chunk {chunk_i + 1}/{len(chunks)}"),
                )

                if (first_error or run_had_fatal_error) and (cfg.fail_fast or quota_exhausted):
                    break

            progress.update(chunk_task, visible=False)
            progress.update(cell_task, visible=False)
            progress.update(p2_task, visible=False)

            # Update run bar to show outcome
            if run_cell_errors > 0:
                progress.update(
                    run_task,
                    description=f"[red]✗ Run {run.run_id} ({run.name}) — {run_cell_errors} cell(s) failed[/red]",
                )
            else:
                progress.update(
                    run_task,
                    description=f"[green]✓ Run {run.run_id} ({run.name})[/green]",
                )

            progress.advance(run_task)
            logger.debug("Run %d complete", run.run_id)

            run_failed = (bool(first_error) or run_had_fatal_error) and (cfg.fail_fast or quota_exhausted)
            any_run_failed = any_run_failed or run_failed
            will_continue = (run_idx < len(runs) - 1) and not run_failed

            # Let any still-in-flight per-cell heartbeats land before this run's own
            # terminal signal, so a late one can never race past it and revert the
            # job's reported status.
            if _heartbeat_tasks:
                await asyncio.gather(*_heartbeat_tasks, return_exceptions=True)

            # Multi-run pipelines (a whole ExperimentSetup run sequentially, see
            # build_setup_pipeline_json on the SEER side) share one --progress-url/
            # AnnotationJob across every run they process: only the LAST run actually
            # processed (either the literal last one, or the run a fail_fast abort stops
            # on) may report a terminal "succeeded"/"failed" status. An earlier run
            # finishing successfully must still report "running", or SEER's poller would
            # think the whole job is done after run 1 of N and stop watching before the
            # rest even start.
            if will_continue:
                await reporter.heartbeat(
                    status="running",
                    cells_total=cells_total,
                    cells_done=cell_counters["done"],
                    cells_error=cell_counters["error"],
                    message=_progress_message("done, moving to next run"),
                )
            else:
                await reporter.heartbeat(
                    status="failed" if run_failed else "succeeded",
                    cells_total=cells_total,
                    cells_done=cell_counters["done"],
                    cells_error=cell_counters["error"],
                    message=_progress_message("failed" if run_failed else "succeeded"),
                )

            if run_failed:
                break

    if all_run_errors:
        _console.print("\n[bold red]Errors:[/bold red]")
        for msg in all_run_errors:
            _console.print(f"  [red]•[/red] {msg}")

    if first_error:
        raise first_error[0]
    if any_run_failed:
        # Carry the reasons in the exception, not only in the console/log. The
        # caller may be a worker with no terminal attached (SEER's llm_runner),
        # and this message is what it stores on the job row — "see errors above"
        # points at output that reader never had.
        if all_run_errors:
            raise RuntimeError("One or more runs failed: " + " | ".join(all_run_errors))
        raise RuntimeError("One or more runs failed — see errors above.")


async def pass1_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
) -> tuple[int, int]:
    """Run only Pass-1 for all pending cells, saving results as pass1_done.

    Returns (n_pass1_cells_saved, n_failed).
    """
    from .seer_client import DryRunSeerClient

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)
    if concurrency is not None:
        pipeline_cfg.concurrency = concurrency
    if rpm is not None:
        pipeline_cfg.per_provider_rpm = rpm

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params
    _litellm.suppress_debug_info = True

    store = Store(settings.runtime.store_path)
    client = (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    sem = asyncio.Semaphore(pipeline_cfg.concurrency)
    limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]

    n_saved = 0
    n_failed = 0

    for run in runs:
        cfg = effective_run_config(run.config, settings.run_defaults)
        _litellm.drop_params = cfg.drop_params
        groups_def = resolve_groups(cfg, pipeline.questions)

        # Resolve source texts for all papers
        source_texts: dict[int, str] = {}
        for paper in papers:
            if cfg.text_source == "full_text":
                source = await full_text_for_run(
                    client, store, paper.paper_id, run.run_id,
                )
                if source is not None:
                    source_texts[paper.paper_id] = source
                # papers without OCR are skipped silently for pass1
            else:
                source_texts[paper.paper_id] = paper.abstract

        # Build pending_cells, but pre-filter to exclude pass1_done/done/posted
        # _execute_pass1 uses should_skip_cell (done/posted only) — we additionally
        # exclude pass1_done by providing a filtered source_texts that omits those papers.
        # Strategy: for each candidate (paper, group), check if all questions in that
        # group are already pass1_done/done/posted; if so, exclude from source_texts
        # for _execute_pass1 by using a filtered set.
        # Actually the cleanest approach: let _execute_pass1 build pending_cells normally,
        # then remove cells that are pass1_done afterward (before saving).

        pending_cells: dict[str, tuple] = {}
        # p1_errors (per-cid detail) isn't wired into this standalone pass1-only
        # entrypoint's n_failed accounting below (out of scope for this pass — it
        # already infers failure from a missing cid in p1_texts); kept for the
        # 4-tuple return shape shared with run_pipeline.
        p1_texts, p1_usage, p1_errors, p1_error = await _execute_pass1(
            run, cfg, papers, source_texts, pending_cells,
            store, settings, dry_run, groups_def=groups_def,
            sem=sem, limiter=limiter,
        )
        if p1_error:
            logger.error("Pass-1 aborted for run %d: %s", run.run_id, p1_error)

        # Remove cells that are already pass1_done (should_skip_cell only checks done/posted)
        pass1_done_cids = set()
        for cid, (paper, group, group_idx) in list(pending_cells.items()):
            if all(
                store.get_status(run.run_id, paper.paper_id, q.version_id) == "pass1_done"
                for q in group
            ):
                pass1_done_cids.add(cid)
        for cid in pass1_done_cids:
            del pending_cells[cid]

        # Save pass1_done rows for the remaining pending cells
        question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}

        for cid, (paper, group, group_idx) in pending_cells.items():
            p1_text = p1_texts.get(cid)
            if p1_text is None:
                # P1 failed for this cell
                n_failed += len(group)
                continue

            u = p1_usage.get(cid, {})
            tok_input  = u.get("input_tokens", 0)
            tok_output = u.get("output_tokens", 0)
            tok_cached = u.get("cache_read_tokens", 0)
            p1_cost    = u.get("cost")
            p1_latency = u.get("latency_ms", 0) or 0

            for i, question in enumerate(group):
                raw_response: dict = {
                    "text_source": cfg.text_source,
                    "batch_group_id": cid,
                    "batch_mode": cfg.batch_p1,
                    "p1_usage": u,
                }
                if i == 0:
                    raw_response["pass1_text"] = p1_text

                payload = build_llm_answer(
                    run_id=run.run_id,
                    paper_id=paper.paper_id,
                    question=question,
                    value=None,
                    comment="",
                    cited_text="",
                    cited_text_verified=None,
                    raw_response=raw_response,
                    latency_ms=p1_latency if i == 0 else None,
                    tokens_total=tok_input + tok_output + tok_cached if i == 0 else None,
                    tokens_input=tok_input if i == 0 else None,
                    tokens_output=tok_output if i == 0 else None,
                    tokens_cached=tok_cached if i == 0 else None,
                    cost=p1_cost if i == 0 else None,
                    cost_currency="USD",
                )
                store.save_pass1(run.run_id, paper.paper_id, question.version_id, payload, cid)
                n_saved += 1

    logger.info("Pass-1 complete: %d cells saved, %d failed", n_saved, n_failed)
    return n_saved, n_failed


def _pass1_cells_from_store(
    store: Store, run_id: int, paper_id: int,
    question_map: dict, question_order: dict,
) -> list[dict]:
    """The store's ``pass1_done`` rows for one (run, paper), as Pass-1 cells.

    Only `pass1_pipeline` ever writes those rows. A host application that keeps
    its own Pass-1 record has none here to read, which is why `pass2_pipeline`
    also accepts cells of this shape straight from the caller (``pass1_cells=``).

    One cell per question group: ``group_id``, the ``version_ids`` it covers in
    pipeline order, the Pass-1 text, and the Pass-1 usage/cost figures. Those
    last two live only on the group's first row (the ``i == 0`` one); the rest of
    the group repeats neither.
    """
    rows = store.get_pass1_rows(run_id, paper_id)
    if not rows:
        return []

    groups: dict[str, list[dict]] = {}
    for row in rows:
        raw_resp = json.loads(row["payload"]["raw_response"])
        group_id = raw_resp.get("batch_group_id") or f"solo_{row['version_id']}"
        groups.setdefault(group_id, []).append(row)

    cells: list[dict] = []
    for group_id, group_rows in groups.items():
        questions_in_group = sorted(
            [question_map[r["version_id"]] for r in group_rows if r["version_id"] in question_map],
            key=lambda q: question_order.get(q.version_id, 0),
        )
        if not questions_in_group:
            continue
        q_to_row = {r["version_id"]: r for r in group_rows}
        ordered_rows = [q_to_row[q.version_id] for q in questions_in_group if q.version_id in q_to_row]
        if not ordered_rows:
            continue
        raw0 = json.loads(ordered_rows[0]["payload"]["raw_response"])
        cells.append({
            "group_id": group_id,
            "version_ids": [q.version_id for q in questions_in_group],
            "pass1_text": raw0.get("pass1_text") or "",
            "p1_usage": raw0.get("p1_usage") or {},
            "p1_payload": ordered_rows[0]["payload"],
        })
    return cells


def _resolve_pass1_cells(
    cells: list[dict], paper, question_map: dict, question_order: dict,
) -> tuple[dict, dict, dict, dict, int]:
    """Pass-1 cells -> the four per-group maps Pass 2 runs from.

    Returns ``(pending_cells, pending_p1, p1_usage_by_cid, p1_payload_by_cid,
    n_dropped)``. A cell whose Pass-1 text is missing or empty is dropped and
    counted rather than sent: Pass 2 would have nothing to reformat, and asking
    it anyway is an invitation to invent an answer.
    """
    pending_cells: dict[str, tuple] = {}
    pending_p1: dict[str, str] = {}
    p1_usage_by_cid: dict[str, dict] = {}
    p1_payload_by_cid: dict[str, dict] = {}
    n_dropped = 0

    for cell in cells:
        group_id = cell["group_id"]
        questions_in_group = sorted(
            [question_map[v] for v in (cell.get("version_ids") or []) if v in question_map],
            key=lambda q: question_order.get(q.version_id, 0),
        )
        if not questions_in_group:
            continue

        pass1_text = cell.get("pass1_text")
        if not pass1_text:
            logger.warning(
                "No pass1_text for paper %d group %s - skipping %d question(s)",
                paper.paper_id, group_id, len(questions_in_group),
            )
            n_dropped += len(questions_in_group)
            continue

        pending_cells[group_id] = (paper, questions_in_group, 0)
        pending_p1[group_id] = pass1_text
        p1_usage_by_cid[group_id] = cell.get("p1_usage") or {}
        p1_payload_by_cid[group_id] = cell.get("p1_payload") or {}

    return pending_cells, pending_p1, p1_usage_by_cid, p1_payload_by_cid, n_dropped


async def pass2_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
    format_model: str | None = None,
    format_model_provider: str | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
    post: bool = True,
    store: Store | None = None,
    client: SeerClient | None = None,
    pass1_cells: "dict[tuple[int, int], list[dict]] | None" = None,
) -> tuple[int, int]:
    """Run only Pass-2 over already-written Pass-1 text, producing done answers.

    Where the Pass-1 text comes from is the caller's choice:

    - ``pass1_cells=None`` (default) reads the local store's ``pass1_done`` rows,
      which only a standalone `pass1_pipeline` invocation ever writes. After a
      normal end-to-end `run_pipeline` there are none, and this returns (0, 0).
    - ``pass1_cells={(run_id, paper_id): [cell, ...]}`` uses what the caller
      passes in - see `_pass1_cells_from_store` for the cell shape. This is how a
      host application that keeps its own Pass-1 record retries a failed Pass 2
      without paying for Pass 1 again.

    ``store`` and ``client`` are injectable for the same reason they are on
    `run_pipeline`: an in-process host writes answers through its own client
    rather than over HTTP.

    ``format_model`` / ``format_model_provider`` override the Pass-2 target for
    every run. Retrying a bad endpoint against itself mostly fails, so being able
    to send the retry somewhere else is the point of the override.

    Returns (n_done, n_failed).
    """
    from .seer_client import DryRunSeerClient

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)
    if concurrency is not None:
        pipeline_cfg.concurrency = concurrency
    if rpm is not None:
        pipeline_cfg.per_provider_rpm = rpm

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params
    _litellm.suppress_debug_info = True

    store = store or Store(settings.runtime.store_path)
    client = client or (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    sem = asyncio.Semaphore(pipeline_cfg.concurrency)
    limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]

    question_map = {q.version_id: q for q in pipeline.questions}
    question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}

    n_done = 0
    n_failed = 0

    for run in runs:
        cfg = effective_run_config(run.config, settings.run_defaults)
        _litellm.drop_params = cfg.drop_params

        # Apply format model overrides
        if format_model is not None:
            cfg.format_model = format_model
        if format_model_provider is not None:
            cfg.format_model_provider = format_model_provider

        for paper in papers:
            if pass1_cells is None:
                cells = _pass1_cells_from_store(
                    store, run.run_id, paper.paper_id, question_map, question_order,
                )
            else:
                cells = pass1_cells.get((run.run_id, paper.paper_id)) or []
            if not cells:
                continue

            # Resolve source text for citation verification. A caller-supplied
            # store holds no OCR — only a Pass-1 run through this pipeline puts it
            # there — so fetch it the way run_pipeline does. Verifying quotes
            # against an empty string does not fail loudly; it silently reports
            # every citation in the paper as unverified.
            if cfg.text_source == "full_text":
                source_text = await full_text_for_run(
                    client, store, paper.paper_id, run.run_id,
                ) or ""
            else:
                source_text = paper.abstract

            (
                pending_cells,
                pending_p1,
                p1_usage_by_cid,
                p1_payload_by_cid,
                n_no_pass1,
            ) = _resolve_pass1_cells(cells, paper, question_map, question_order)
            n_failed += n_no_pass1

            if not pending_p1:
                continue

            # source_texts dict for tail logic
            source_texts = {paper.paper_id: source_text}

            # Run pass 2 (p2_errors unused here — this standalone pass2-only
            # entrypoint already infers failure from a missing cid in p2_texts;
            # kept for the 3-tuple return shape shared with run_pipeline)
            p2_texts, p2_usage, p2_errors = await _execute_pass2(
                run, cfg, pending_p1, pending_cells,
                store, settings, dry_run,
                sem=sem, limiter=limiter,
                response_format=ANNOTATE_RESPONSE_FORMAT,
                require_status=True,
            )

            # Parse / verify / save with p1 token carry-over
            for cid, p2_text in p2_texts.items():
                if cid not in pending_cells:
                    continue
                paper_cell, group, group_idx = pending_cells[cid]
                p1_payload_row0 = p1_payload_by_cid.get(cid, {})
                u = p1_usage_by_cid.get(cid, {})

                # p1 token/cost figures from stored payload (row 0)
                p1_tok_input  = p1_payload_row0.get("tokens_input")
                p1_tok_output = p1_payload_row0.get("tokens_output")
                p1_tok_cached = p1_payload_row0.get("tokens_cached")
                p1_tok_total  = p1_payload_row0.get("tokens_total")
                p1_cost_str   = p1_payload_row0.get("cost")
                p1_cost       = Decimal(p1_cost_str) if p1_cost_str else None
                p1_latency    = p1_payload_row0.get("latency_ms", 0) or 0

                # p2 (format) figures from this stage's usage
                fu = p2_usage.get(cid, {})
                fmt_input  = fu.get("input_tokens", 0)
                fmt_output = fu.get("output_tokens", 0)
                fmt_cached = fu.get("cache_read_tokens", 0)
                fmt_total  = fmt_input + fmt_output + fmt_cached
                fmt_cost   = fu.get("cost")
                p2_latency = fu.get("latency_ms", 0) or 0

                # Same reply-completeness gate as the run_pipeline tail above.
                # It matters at least as much here: this is the path a Pass-2
                # retry takes, and a retry sent back to an endpoint that is
                # still looping would otherwise overwrite the failed cell with a
                # fresh answer read off a fresh loop.
                incomplete = pass2_incomplete_reason(p2_text, fu)
                if incomplete is not None:
                    code, detail = incomplete
                    logger.error(
                        "EXTRACTION FAILURE (pass2) — run %d, paper %d, group %s: %s",
                        run.run_id, paper_cell.paper_id, cid, detail,
                    )
                    for question in group:
                        store.save_answer(
                            run.run_id, paper_cell.paper_id, question.version_id,
                            build_error_answer(
                                run_id=run.run_id, paper_id=paper_cell.paper_id,
                                question=question, extraction_detail=detail,
                                raw_response={
                                    "pass1_text": pending_p1.get(cid, ""),
                                    "pass2_text": p2_text,
                                    "p1_usage": u,
                                    "p2_usage": fu,
                                    "diagnostics": [
                                        {"phase": "pass2", "code": code, "detail": detail}
                                    ],
                                    "batch_group_id": cid,
                                },
                            ),
                            cid,
                        )
                        n_failed += 1
                    continue

                parsed = parse_structured_output(
                    p2_text, [q.key for q in group], annotate_mode=True, require_status=True,
                )
                parsed, _excl_idx = apply_scope_and_status(
                    group, parsed, enabled=cfg.early_exit_on_ic_exclusion,
                    # .get(cid) -> None when absent, NOT "": an empty string would
                    # read as "pass-1 answered nothing" and demote the entire group
                    # to absent. None skips the check, failing toward prior behavior.
                    pass1_text=pending_p1.get(cid),
                )

                for i, (question, result) in enumerate(zip(group, parsed)):
                    status = result.get("extraction_status", "ok")

                    if status == "skipped":
                        store.save_answer(
                            run.run_id, paper_cell.paper_id, question.version_id,
                            build_skipped_answer(
                                run_id=run.run_id, paper_id=paper_cell.paper_id, question=question,
                                extraction_detail=result.get("extraction_detail", "gated out"),
                            ),
                            cid,
                        )
                        n_done += 1
                        continue

                    if status == "absent":
                        # Prefer the demotion's own extraction_detail (e.g. "pass-1
                        # has no answer block for ...") over the generic message.
                        err_detail = result.get(
                            "extraction_detail", f"key {question.key!r} not found in pass-2 output"
                        )
                        logger.error(
                            "EXTRACTION FAILURE (pass2) — run %d, paper %d, group %s, key %r: %s",
                            run.run_id, paper_cell.paper_id, cid, question.key, err_detail,
                        )
                        store.save_answer(
                            run.run_id, paper_cell.paper_id, question.version_id,
                            build_error_answer(run_id=run.run_id, paper_id=paper_cell.paper_id, question=question, extraction_detail=err_detail),
                            cid,
                        )
                        n_failed += 1
                        continue

                    # Either way the item is not what the model wrote: the first
                    # is damage that landed inside cited_text, the second is an
                    # item the reply stopped partway through (see
                    # annotate.parse.cited_text_violation and
                    # _incomplete_item_violation).
                    item_error = result.get("cited_text_error") or result.get("item_error")
                    if item_error:
                        logger.error(
                            "EXTRACTION FAILURE (pass2) — run %d, paper %d, group %s, key %r: %s",
                            run.run_id, paper_cell.paper_id, cid, question.key, item_error,
                        )
                        store.save_answer(
                            run.run_id, paper_cell.paper_id, question.version_id,
                            build_error_answer(
                                run_id=run.run_id, paper_id=paper_cell.paper_id,
                                question=question, extraction_detail=item_error,
                            ),
                            cid,
                        )
                        n_failed += 1
                        continue

                    verify = verify_citation(
                        result.get("cited_text", ""),
                        source_text,
                        max_error_rate=cfg.citation_max_error_rate,
                        max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                    )
                    cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]

                    verify_list = verify_citations(
                        result.get("cited_text", ""),
                        source_text,
                        max_error_rate=cfg.citation_max_error_rate,
                        max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                    )
                    citations = build_citations(result.get("cited_text", ""), verify_list)

                    # Retrieve pass1_text for raw_response from pending_p1
                    p1_text_for_raw = pending_p1.get(cid, "")

                    raw_response = {
                        "pass1_text": p1_text_for_raw,
                        "pass2_text": p2_text,
                        "parse_result": result,
                        "verify": verify,
                        "text_source": cfg.text_source,
                        "batch_group_id": cid,
                        "batch_mode": True,
                        "p1_usage": u,
                    }

                    status_kwargs: dict = {}
                    if status == "invalid":
                        status_kwargs = {
                            "extraction_status": "invalid",
                            "extraction_detail": f"pass-2 reported unmappable value for {question.key!r}",
                        }

                    payload = build_llm_answer(
                        run_id=run.run_id,
                        paper_id=paper_cell.paper_id,
                        question=question,
                        value=result.get("value"),
                        comment=result.get("comment", ""),
                        cited_text=result.get("cited_text", ""),
                        cited_text_verified=cited_text_verified,
                        citations=citations,
                        raw_response=raw_response,
                        latency_ms=(p1_latency + p2_latency) if i == 0 else None,
                        # Carry p1 token/cost from stored pass1 payload (i==0 only)
                        tokens_total=p1_tok_total if i == 0 else None,
                        tokens_input=p1_tok_input if i == 0 else None,
                        tokens_output=p1_tok_output if i == 0 else None,
                        tokens_cached=p1_tok_cached if i == 0 else None,
                        cost=p1_cost if i == 0 else None,
                        cost_currency="USD",
                        # p2 (format) figures from this stage (i==0 only)
                        fmt_tokens_total=fmt_total if i == 0 else None,
                        fmt_tokens_input=fmt_input if i == 0 else None,
                        fmt_tokens_output=fmt_output if i == 0 else None,
                        fmt_tokens_cached=fmt_cached if i == 0 else None,
                        fmt_cost=fmt_cost if i == 0 else None,
                        confidence=result.get("confidence"),
                        **status_kwargs,
                    )
                    store.save_answer(run.run_id, paper_cell.paper_id, question.version_id, payload, cid)
                    n_done += 1

            # Post per paper if requested
            if post:
                unposted = store.get_unposted(run.run_id, paper.paper_id)
                if unposted:
                    try:
                        await client.post_answers_bulk(unposted)
                        if not dry_run:
                            version_ids = [p["question_version"] for p in unposted]
                            store.mark_posted(run.run_id, paper.paper_id, version_ids)
                        logger.info("Posted %d answers for paper %d", len(unposted), paper.paper_id)
                    except Exception as exc:
                        logger.error("Post failed for paper %d: %s", paper.paper_id, exc)

    logger.info("Pass-2 complete: %d done, %d failed", n_done, n_failed)
    return n_done, n_failed


async def reformat_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
    client: SeerClient | None = None,
    format_model: str | None = None,
    format_model_provider: str | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
) -> tuple[int, int]:
    """Re-run pass-2 formatting on stored answers, optionally with a different model.

    Returns (n_updated, n_failed).
    Answers are reset to status='done' so they can be reposted afterwards.
    """
    import asyncio
    import functools

    from .annotate.engine import reformat_group
    from .annotate.parse import ExtractionError
    from .llm import complete as llm_complete, dummy_complete

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params
    _litellm.suppress_debug_info = True

    store = Store(settings.runtime.store_path)
    # Nothing is fetched here — the client is asked only which rendering of a
    # paper the run was sent, so the stored text is read back under the right
    # key. See `reformat_arbitration_pipeline` for the same argument.
    client = client or SeerClient(
        pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions,
    )
    question_map = {q.version_id: q for q in pipeline.questions}
    question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}
    paper_map = {p.paper_id: p for p in pipeline.papers}

    _base_complete = dummy_complete if dry_run else llm_complete
    _limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    async def _rate_limited(model, provider, messages, **kw):
        await _limiter.acquire(provider)
        return await _base_complete(model, provider, messages, **kw)

    sem = asyncio.Semaphore(pipeline_cfg.concurrency)

    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]

    total_done = 0
    total_failed = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

    with progress:
        run_task = progress.add_task("Runs", total=len(runs))
        paper_task = progress.add_task("", total=len(papers), visible=False)

        for run in runs:
            progress.update(run_task, description=f"Run {run.run_id} [cyan]({run.name})[/cyan]")
            progress.reset(paper_task, total=len(papers), visible=True)

            cfg = effective_run_config(run.config, settings.run_defaults)
            _litellm.drop_params = cfg.drop_params
            # CLI arg > effective run config (run_defaults merged with per-run) > run's own provider
            final_model = format_model or cfg.format_model or "gpt-4o-mini"
            final_provider = (
                format_model_provider
                or cfg.format_model_provider
                or run.model_provider
            )

            fmt_extra: dict = {}
            if not dry_run:
                fmt_prov = settings.providers.get(final_provider, ProviderSettings())
                api_key = fmt_prov.resolved_api_key()
                if api_key:
                    fmt_extra["api_key"] = api_key
                if fmt_prov.base_url:
                    fmt_extra["api_base"] = fmt_prov.base_url
                if fmt_prov.api_version:
                    fmt_extra["api_version"] = fmt_prov.api_version
            fmt_complete_fn = (
                functools.partial(_rate_limited, **fmt_extra) if fmt_extra else _rate_limited
            )

            for paper in papers:
                progress.update(paper_task, description=f"  Paper {paper.paper_id}")

                rows = store.get_reformattable_rows(run.run_id, paper.paper_id)
                if not rows:
                    progress.advance(paper_task)
                    continue

                if cfg.text_source == "full_text":
                    source_text = cached_full_text_for_run(
                        client, store, paper.paper_id, run.run_id,
                    ) or ""
                else:
                    source_text = paper_map.get(paper.paper_id, paper).abstract

                # Group by batch_group_id so multi-question groups are reformatted together
                groups: dict[str, list[dict]] = {}
                for row in rows:
                    raw_resp = json.loads(row["payload"]["raw_response"])
                    group_id = raw_resp.get("batch_group_id") or f"solo_{row['version_id']}"
                    groups.setdefault(group_id, []).append(row)

                async def _do_group(
                    group_id: str,
                    group_rows: list[dict],
                    run=run,
                    paper=paper,
                    source_text=source_text,
                    final_model=final_model,
                    final_provider=final_provider,
                    fmt_complete_fn=fmt_complete_fn,
                ) -> None:
                    nonlocal total_done, total_failed

                    raw0 = json.loads(group_rows[0]["payload"]["raw_response"])
                    pass1_text = raw0.get("pass1_text")
                    if not pass1_text:
                        logger.warning(
                            "No pass1_text in raw_response for run=%d paper=%d group=%s — skipping",
                            run.run_id, paper.paper_id, group_id,
                        )
                        total_failed += len(group_rows)
                        return

                    # Sort questions by original pipeline order so the format prompt is stable
                    questions_in_group = sorted(
                        [question_map[r["version_id"]] for r in group_rows if r["version_id"] in question_map],
                        key=lambda q: question_order.get(q.version_id, 0),
                    )
                    q_to_row = {r["version_id"]: r for r in group_rows}
                    ordered_rows = [q_to_row[q.version_id] for q in questions_in_group]

                    if dry_run:
                        logger.info(
                            "Dry-run: run=%d paper=%d group=%s (%d q) model=%s/%s",
                            run.run_id, paper.paper_id, group_id,
                            len(questions_in_group), final_provider, final_model,
                        )
                        return

                    async with sem:
                        try:
                            results = await reformat_group(
                                pass1_text=pass1_text,
                                source_text=source_text,
                                questions=questions_in_group,
                                format_model=final_model,
                                format_model_provider=final_provider,
                                format_structured_output=cfg.format_structured_output,
                                format_temperature=cfg.format_temperature,
                                format_model_params=cfg.format_model_params,
                                complete_fn=fmt_complete_fn,
                                citation_max_error_rate=cfg.citation_max_error_rate,
                                citation_max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                                request_timeout=cfg.request_timeout,
                            )
                        except ExtractionError as exc:
                            logger.error(
                                "Reformat ExtractionError run=%d paper=%d group=%s: %s",
                                run.run_id, paper.paper_id, group_id, exc,
                            )
                            total_failed += len(group_rows)
                            return
                        except Exception as exc:
                            logger.error(
                                "Reformat failed run=%d paper=%d group=%s: %s",
                                run.run_id, paper.paper_id, group_id, exc,
                            )
                            total_failed += len(group_rows)
                            return

                    for i, (row, result) in enumerate(zip(ordered_rows, results)):
                        old_payload = row["payload"]
                        old_raw = json.loads(old_payload["raw_response"])
                        new_raw = {
                            **old_raw,
                            "parse_result": result["parse_result"],
                            "verify": result["verify"],
                        }
                        if i == 0:
                            new_raw["pass2_text"] = result["p2_text"]
                            new_raw["p2_raw"] = result["p2_raw"]
                        question = question_map[row["version_id"]]
                        parsed = result["parse_result"]
                        new_payload = build_llm_answer(
                            run_id=row["run_id"],
                            paper_id=row["paper_id"],
                            question=question,
                            value=parsed.get("value"),
                            comment=parsed.get("comment", ""),
                            cited_text=parsed.get("cited_text", ""),
                            cited_text_verified=result["cited_text_verified"],
                            citations=result["citations"],
                            raw_response=new_raw,
                            latency_ms=old_payload.get("latency_ms"),
                            tokens_total=old_payload.get("tokens_total"),
                            tokens_input=old_payload.get("tokens_input"),
                            tokens_output=old_payload.get("tokens_output"),
                            tokens_cached=old_payload.get("tokens_cached"),
                            cost=Decimal(old_payload["cost"]) if old_payload.get("cost") else None,
                            cost_currency=old_payload.get("cost_currency", "USD"),
                            fmt_tokens_total=result["fmt_tokens_total"],
                            fmt_tokens_input=result["fmt_tokens_input"],
                            fmt_tokens_output=result["fmt_tokens_output"],
                            fmt_tokens_cached=result["fmt_tokens_cached"],
                            fmt_cost=result["fmt_cost"],
                            confidence=parsed.get("confidence"),
                        )
                        store.update_reformatted(row["run_id"], row["paper_id"], row["version_id"], new_payload)
                        total_done += 1

                await asyncio.gather(*[_do_group(gid, grows) for gid, grows in groups.items()])

                progress.advance(paper_task)

            progress.advance(run_task)

    logger.info("Reformat complete: %d updated, %d failed", total_done, total_failed)
    return total_done, total_failed
