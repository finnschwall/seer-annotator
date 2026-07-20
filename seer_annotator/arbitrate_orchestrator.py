"""Arbitration loop: dispute → adjudicate → format → verify → persist → post.

Mirrors orchestrator.py's structure function-for-function, reusing the same
_execute_pass1/_execute_pass2 engine (batch_runner.py) and Pass-2 machinery
(annotate.parse/annotate.verify/annotate.engine.reformat_group) — only Pass-1
message construction (arbitrate.prompt) and the tail payload builder
(mapping.build_resolution) are arbitration-specific. This includes the
--progress-url heartbeat wiring in run_arbitration_pipeline, which mirrors
run_pipeline's (see orchestrator.py) with one cell == one dispute item instead
of one paper x question.
"""

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
from .batch_runner import _execute_pass1, _execute_pass2
from .config import (
    ArbiterRunConfig,
    Candidate,
    DisputeItem,
    DisputePipelineConfig,
    Paper,
    ProviderSettings,
    Question,
    Settings,
    effective_arbiter_config,
)
from .progress import ProgressReporter, ProgressReporterProtocol
from .rate_limiter import PerProviderRateLimiter
from .seer_client import SeerClient
from .store import Store
from .mapping import build_error_resolution, build_resolution
from .annotate.parse import ExtractionError, parse_structured_output
from .annotate.verify import verify_citation
from .arbitrate.prompt import build_dispute_messages
from .orchestrator import _chunks

logger = logging.getLogger(__name__)


def _papers_from_disputes(disputes: list[DisputeItem]) -> list[Paper]:
    """Dispute pipeline JSON has no papers[] — synthesize one Paper per distinct
    paper appearing in disputes[], reusing config.Paper (paper_id/title/abstract)."""
    seen: dict[int, Paper] = {}
    for d in disputes:
        if d.paper_id not in seen:
            seen[d.paper_id] = Paper(paper_id=d.paper_id, title=d.paper_title, abstract=d.abstract, split="")
    return list(seen.values())


def _dispute_index(
    disputes: list[DisputeItem],
) -> tuple[dict[tuple[int, int], DisputeItem], dict[tuple[int, int], list[Candidate]]]:
    """Returns (dispute_by_paper_version, candidates_by_paper_version), both keyed
    by (paper_id, version_id) — the unique locator for a dispute within one dispute set."""
    dispute_by_paper_version: dict[tuple[int, int], DisputeItem] = {}
    candidates_by_paper_version: dict[tuple[int, int], list[Candidate]] = {}
    for d in disputes:
        key = (d.paper_id, d.version_id)
        dispute_by_paper_version[key] = d
        candidates_by_paper_version[key] = d.candidates
    return dispute_by_paper_version, candidates_by_paper_version


def _questions_for_paper(disputes_for_paper: list[DisputeItem], question_by_key: dict[str, Question]) -> list[Question]:
    questions: list[Question] = []
    seen_versions: set[int] = set()
    for d in disputes_for_paper:
        q = question_by_key.get(d.question_key)
        if q is None or q.version_id in seen_versions:
            continue
        seen_versions.add(q.version_id)
        questions.append(q)
    return questions


def _build_p1_messages_fn(cfg: ArbiterRunConfig, candidates_by_paper_version: dict[tuple[int, int], list[Candidate]]):
    """Closure passed as batch_runner's build_p1_messages — (source, group, paper_id) -> messages."""

    def _fn(source_text: str, group: list[Question], paper_id: int) -> list[dict]:
        candidates_by_version_id = {
            q.version_id: candidates_by_paper_version.get((paper_id, q.version_id), []) for q in group
        }
        return build_dispute_messages(
            None if cfg.text_source == "candidates_only" else source_text,
            group,
            candidates_by_version_id,
            anonymize_raters=cfg.anonymize_raters,
            text_source=cfg.text_source,
            system_prompt=cfg.system_prompt,
            cache_first=cfg.cache_first,
        )

    return _fn


def _skip_resolution_cell_fn(store: Store):
    """Closure passed as batch_runner's should_skip_cell — checks the resolutions table."""

    def _fn(run_id: int, paper_id: int, version_id: int) -> bool:
        return store.should_skip_resolution_cell_by_paper_version(run_id, paper_id, version_id)

    return _fn


def _resolve_source_text(cfg: ArbiterRunConfig, paper: Paper, store: Store) -> str:
    if cfg.text_source == "candidates_only":
        return ""
    if cfg.text_source == "abstract":
        return paper.abstract
    return store.get_ocr(paper.paper_id) or ""


def _parse_save_post_tail_resolutions(
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
    dispute_by_paper_version: dict[tuple[int, int], DisputeItem],
    fail_fast: bool,
    on_payload_saved: "Callable[[dict], None] | None" = None,
) -> "ExtractionError | None":
    """Parse/verify/save resolutions from p2_texts. Returns first ExtractionError if any.

    ``on_payload_saved``, when given, is called once per built payload right after
    it's persisted — used by the progress heartbeat to track cells_done/cells_error
    without a separate store query (mirrors _parse_save_post_tail in orchestrator.py).
    """
    first_extraction_error: ExtractionError | None = None

    for cid in p2_texts:
        if cid not in pending_cells:
            continue
        paper, group, group_idx = pending_cells[cid]
        p1_text = p1_texts.get(cid, "")
        p2_text = p2_texts[cid]

        u = p1_usage.get(cid, {})
        tok_input = u.get("input_tokens", 0)
        tok_output = u.get("output_tokens", 0)
        tok_cached = u.get("cache_read_tokens", 0)
        p1_cost = u.get("cost")
        p1_latency = u.get("latency_ms", 0) or 0

        fu = p2_usage.get(cid, {})
        fmt_input = fu.get("input_tokens", 0)
        fmt_output = fu.get("output_tokens", 0)
        fmt_cached = fu.get("cache_read_tokens", 0)
        fmt_total = fmt_input + fmt_output + fmt_cached
        fmt_cost = fu.get("cost")
        p2_latency = fu.get("latency_ms", 0) or 0

        parsed = parse_structured_output(p2_text, [q.key for q in group])

        failed_keys = {r["key"]: r["parse_error"] for r in parsed if "parse_error" in r}
        if failed_keys:
            err = ExtractionError(failed_keys)
            logger.error(
                "EXTRACTION FAILURE — run %d, paper %d, group %d: %s\n"
                "  Pass-2 output was:\n%s",
                run.run_id, paper.paper_id, group_idx, err,
                p2_text[:2000],
            )
            for q in group:
                d = dispute_by_paper_version.get((paper.paper_id, q.version_id))
                if d is None:
                    continue
                error_payload = build_error_resolution(
                    arbiter_run_id=run.run_id, paper_id=paper.paper_id,
                    dispute_item_id=d.dispute_item_id, question=q, resolution_detail=str(err),
                )
                store.save_resolution(
                    run.run_id, d.dispute_item_id, paper.paper_id, q.version_id, error_payload,
                )
                if on_payload_saved is not None:
                    on_payload_saved(error_payload)
            if fail_fast and first_extraction_error is None:
                first_extraction_error = err
            continue

        for i, (question, result) in enumerate(zip(group, parsed)):
            d = dispute_by_paper_version.get((paper.paper_id, question.version_id))
            if d is None:
                logger.warning(
                    "No dispute item for paper=%d version_id=%d — skipping",
                    paper.paper_id, question.version_id,
                )
                continue

            verify = verify_citation(
                result.get("cited_text", ""),
                source_texts.get(paper.paper_id, "") or "",
                max_error_rate=cfg.citation_max_error_rate,
                max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
            )
            cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]
            raw_response = {
                "pass1_text": p1_text,
                "pass2_text": p2_text,
                "parse_result": result,
                "verify": verify,
                "text_source": cfg.text_source,
                "batch_group_id": cid,
                "batch_mode": True,
                "p1_usage": u,
            }
            payload = build_resolution(
                arbiter_run_id=run.run_id,
                paper_id=paper.paper_id,
                dispute_item_id=d.dispute_item_id,
                question=question,
                value=result.get("value"),
                comment=result.get("comment", ""),
                cited_text=result.get("cited_text", ""),
                cited_text_verified=cited_text_verified,
                raw_response=raw_response,
                latency_ms=(p1_latency + p2_latency) if i == 0 else 0,
                tokens_total=tok_input + tok_output + tok_cached if i == 0 else 0,
                tokens_input=tok_input if i == 0 else 0,
                tokens_output=tok_output if i == 0 else 0,
                tokens_cached=tok_cached if i == 0 else 0,
                cost=p1_cost if i == 0 else None,
                cost_currency="USD",
                fmt_tokens_total=fmt_total if i == 0 else 0,
                fmt_tokens_input=fmt_input if i == 0 else 0,
                fmt_tokens_output=fmt_output if i == 0 else 0,
                fmt_tokens_cached=fmt_cached if i == 0 else 0,
                fmt_cost=fmt_cost if i == 0 else None,
                confidence=result.get("confidence"),
            )
            store.save_resolution(run.run_id, d.dispute_item_id, paper.paper_id, question.version_id, payload, cid)
            if on_payload_saved is not None:
                on_payload_saved(payload)

    return first_extraction_error


async def run_arbitration_pipeline(
    pipeline: DisputePipelineConfig,
    settings: Settings,
    *,
    store: Store | None = None,
    client: SeerClient | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    dispute_item_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
    progress_url: str | None = None,
    reporter_factory: "Callable[[int], ProgressReporterProtocol] | None" = None,
    should_cancel: "Callable[[], bool] | None" = None,
) -> None:
    from .seer_client import DryRunSeerClient

    pipeline_cfg = effective_arbiter_config(ArbiterRunConfig(), settings.arbiter_run_defaults)
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
    disputes = [d for d in pipeline.disputes if (dispute_item_ids is None or d.dispute_item_id in dispute_item_ids)]

    question_by_key = {q.key: q for q in pipeline.questions}
    dispute_by_paper_version, candidates_by_paper_version = _dispute_index(disputes)
    all_papers = _papers_from_disputes(disputes)
    disputes_by_paper: dict[int, list[DisputeItem]] = {}
    for d in disputes:
        disputes_by_paper.setdefault(d.paper_id, []).append(d)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

    all_run_errors: list[str] = []
    first_error: list[ExtractionError] = []
    any_run_failed = False

    with progress:
        run_task = progress.add_task("Arbiter runs", total=len(runs))
        chunk_task = progress.add_task("  Chunks", total=0, visible=False)
        cell_task = progress.add_task("    Cells", total=0, visible=False)
        p2_task = progress.add_task("    Pass 2", total=0, visible=False)

        for run_idx, run in enumerate(runs):
            progress.update(
                run_task,
                description=f"Run {run.run_id} [cyan]({run.name})[/cyan]",
            )
            logger.debug("Arbiter run %d (%s) starting", run.run_id, run.name)

            cfg = effective_arbiter_config(run.config, settings.arbiter_run_defaults)
            _litellm.drop_params = cfg.drop_params

            groups_def: dict[int, list[list[Question]]] = {
                paper_id: resolve_groups(cfg, _questions_for_paper(paper_disputes, question_by_key))
                for paper_id, paper_disputes in disputes_by_paper.items()
            }

            chunks = _chunks(all_papers, cfg.chunk_papers)
            progress.reset(chunk_task, total=len(chunks), visible=True)

            first_error = []
            run_cell_errors = 0
            run_had_fatal_error = False

            reporter = (
                reporter_factory(run.run_id)
                if reporter_factory is not None
                else ProgressReporter(progress_url, pipeline.api_token, run.run_id)
            )
            # A "cell" for arbitration is one dispute item (one paper x question dispute),
            # not one paper x question the way annotation's run_pipeline counts it — disputes
            # are already the flattened per-question unit, so cells_total is simply len(disputes).
            cells_total = len(disputes)
            cell_counters = {"done": 0, "error": 0}

            # Tracks progress within whichever pass is currently in flight for the
            # active chunk (mirrors orchestrator.run_pipeline's phase_state) — reset
            # per chunk and again at the pass1->pass2 handoff.
            phase_state = {"phase": None, "done": 0, "total": 0}

            # Multi-run pipelines share this same --progress-url across every run, so the
            # message is the only signal that tells the poller which run is active.
            def _progress_message(suffix: str, _idx=run_idx, _run=run) -> str:
                if len(runs) <= 1:
                    return suffix
                return f"run {_idx + 1}/{len(runs)} ({_run.name}): {suffix}"

            # Fire-and-forget per-cell heartbeats, throttled — supplements the guaranteed
            # start/chunk-boundary/end heartbeats below with something closer to real-time
            # progress (mirrors orchestrator.run_pipeline's _maybe_heartbeat).
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

            def _count_payload(payload: dict, _c=cell_counters) -> None:
                _c["done"] += 1
                if payload.get("extraction_status") == "error":
                    _c["error"] += 1
                _maybe_heartbeat()

            await reporter.heartbeat(
                status="running", cells_total=cells_total, cells_done=0, cells_error=0,
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

                source_texts: dict[int, str] = {}
                for paper in chunk_papers:
                    if cfg.text_source == "candidates_only":
                        source_texts[paper.paper_id] = ""
                        continue
                    if cfg.text_source == "full_text":
                        source = store.get_ocr(paper.paper_id)
                        if source is None:
                            source = await client.fetch_ocr_markdown(paper.paper_id)
                            store.save_ocr(paper.paper_id, source)
                        if source is None:
                            logger.warning(
                                "No OCR for paper %d — posting error for all its disputes", paper.paper_id
                            )
                            for d in disputes_by_paper.get(paper.paper_id, []):
                                q = question_by_key.get(d.question_key)
                                if q is None:
                                    continue
                                no_ocr_payload = build_error_resolution(
                                    arbiter_run_id=run.run_id, paper_id=paper.paper_id,
                                    dispute_item_id=d.dispute_item_id, question=q,
                                    resolution_detail="no_ocr",
                                )
                                store.save_resolution(
                                    run.run_id, d.dispute_item_id, paper.paper_id, d.version_id,
                                    no_ocr_payload,
                                )
                                _count_payload(no_ocr_payload)
                        else:
                            source_texts[paper.paper_id] = source
                    else:
                        source_texts[paper.paper_id] = paper.abstract

                pending_cells: dict[str, tuple] = {}
                p1_texts, p1_usage, p1_errors, p1_error = await _execute_pass1(
                    run, cfg, chunk_papers, source_texts, pending_cells,
                    store, settings, dry_run, groups_def=groups_def,
                    chunk_i=chunk_i, sem=sem, limiter=limiter,
                    on_cell_done=_on_cell_done, on_total_known=_on_total,
                    build_p1_messages=_build_p1_messages_fn(cfg, candidates_by_paper_version),
                    should_skip_cell=_skip_resolution_cell_fn(store),
                )

                n_pending = len(pending_cells)
                n_got_p1 = len(p1_texts)
                chunk_failed = n_pending - n_got_p1
                run_cell_errors += chunk_failed

                if p1_error:
                    msg = f"Run {run.run_id} ({run.name}): {p1_error}"
                    all_run_errors.append(msg)
                    logger.error("P1 aborted — %s", msg)
                    # p1_error signals a fatal, run-aborting failure (bad model name/API key,
                    # etc.) rather than a per-cell parsing issue — mirrors run_pipeline's
                    # run_had_fatal_error handling so a totally broken run is reported as
                    # failed instead of silently completing 0/N cells as "succeeded".
                    run_had_fatal_error = True
                elif chunk_failed > 0:
                    logger.warning(
                        "Run %d: %d/%d cells failed in chunk %d",
                        run.run_id, chunk_failed, n_pending, chunk_i + 1,
                    )

                # Disputes that entered pending_cells but got no Pass-1 output at all
                # (API error/timeout inside _execute_pass1) would otherwise vanish
                # silently — mirrors run_pipeline's pass1-drop fix (orchestrator.py).
                # Phase 2 only iterates pending_p1 (cids WITH p1 text) and the
                # parse/save tail only iterates p2_texts, so a dropped cid is never
                # touched again. Post an explicit error resolution for each such
                # dispute now, mirroring the no-OCR error path above.
                for cid, (paper, group, group_idx) in pending_cells.items():
                    if cid in p1_texts:
                        continue
                    # Prefer the real per-item error (batch item errored/canceled/
                    # expired, or an online call exception) when available — mirrors
                    # orchestrator.py's run_pipeline handling of the same gap.
                    cell_error_detail = p1_errors.get(
                        cid, "pass1 failed — no output (API error/timeout); see log"
                    )
                    for q in group:
                        d = dispute_by_paper_version.get((paper.paper_id, q.version_id))
                        if d is None:
                            continue
                        p1_drop_payload = build_error_resolution(
                            arbiter_run_id=run.run_id, paper_id=paper.paper_id,
                            dispute_item_id=d.dispute_item_id, question=q,
                            resolution_detail=cell_error_detail,
                        )
                        store.save_resolution(
                            run.run_id, d.dispute_item_id, paper.paper_id, q.version_id, p1_drop_payload,
                        )
                        _count_payload(p1_drop_payload)
                        if cid in p1_errors:
                            store.mark_resolution_failed(run.run_id, d.dispute_item_id, cell_error_detail)

                pending_p1 = {cid: t for cid, t in p1_texts.items() if cid in pending_cells}

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

                p2_texts, p2_usage, p2_errors = await _execute_pass2(
                    run, cfg, pending_p1, pending_cells,
                    store, settings, dry_run,
                    chunk_i=chunk_i, sem=sem, limiter=limiter,
                    on_p2_start=_on_p2_start,
                    on_p2_advance=_on_p2_advance,
                )

                # Mirrors orchestrator.py's p2-drop handling: cells with a Pass-1
                # result that then failed Pass-2 would otherwise vanish silently.
                for cid, err_detail in p2_errors.items():
                    if cid not in pending_cells:
                        continue
                    paper, group, group_idx = pending_cells[cid]
                    for q in group:
                        d = dispute_by_paper_version.get((paper.paper_id, q.version_id))
                        if d is None:
                            continue
                        p2_drop_payload = build_error_resolution(
                            arbiter_run_id=run.run_id, paper_id=paper.paper_id,
                            dispute_item_id=d.dispute_item_id, question=q,
                            resolution_detail=err_detail,
                        )
                        store.save_resolution(
                            run.run_id, d.dispute_item_id, paper.paper_id, q.version_id, p2_drop_payload,
                        )
                        _count_payload(p2_drop_payload)
                        store.mark_resolution_failed(run.run_id, d.dispute_item_id, err_detail)

                err = _parse_save_post_tail_resolutions(
                    p1_texts=p1_texts,
                    p1_usage=p1_usage,
                    p2_texts=p2_texts,
                    p2_usage=p2_usage,
                    pending_cells=pending_cells,
                    source_texts=source_texts,
                    run=run,
                    cfg=cfg,
                    store=store,
                    dispute_by_paper_version=dispute_by_paper_version,
                    fail_fast=cfg.fail_fast,
                    on_payload_saved=_count_payload,
                )
                if err is not None:
                    first_error.append(err)

                for paper in chunk_papers:
                    unposted = store.get_unposted_resolutions(run.run_id, paper.paper_id)
                    if unposted:
                        try:
                            await client.post_resolutions_bulk(unposted)
                            if not dry_run:
                                dispute_ids = [r["dispute_item"] for r in unposted]
                                store.mark_resolutions_posted(run.run_id, dispute_ids)
                            logger.info("Posted %d resolutions for paper %d", len(unposted), paper.paper_id)
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

                if (first_error or run_had_fatal_error) and cfg.fail_fast:
                    break

            progress.update(chunk_task, visible=False)
            progress.update(cell_task, visible=False)
            progress.update(p2_task, visible=False)

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
            logger.debug("Arbiter run %d complete", run.run_id)

            run_failed = (bool(first_error) or run_had_fatal_error) and cfg.fail_fast
            any_run_failed = any_run_failed or run_failed
            will_continue = (run_idx < len(runs) - 1) and not run_failed

            # Let any still-in-flight per-cell heartbeats land before this run's own
            # terminal signal, so a late one can never race past it and revert the
            # job's reported status.
            if _heartbeat_tasks:
                await asyncio.gather(*_heartbeat_tasks, return_exceptions=True)

            # Multi-run pipelines share one --progress-url/AnnotationJob across every run
            # they process: only the LAST run actually processed (either the literal last
            # one, or the run a fail_fast abort stops on) may report a terminal
            # "succeeded"/"failed" status. An earlier run finishing successfully must still
            # report "running", or SEER's poller would think the whole job is done after
            # run 1 of N and stop watching before the rest even start.
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
        raise RuntimeError("One or more arbiter runs failed — see errors above.")


async def arbitration_pass1_pipeline(
    pipeline: DisputePipelineConfig,
    settings: Settings,
    *,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    dispute_item_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
) -> tuple[int, int]:
    """Run only Pass-1 (adjudication reasoning) for all pending disputes, saving
    results as pass1_done. Returns (n_pass1_cells_saved, n_failed)."""
    from .seer_client import DryRunSeerClient

    pipeline_cfg = effective_arbiter_config(ArbiterRunConfig(), settings.arbiter_run_defaults)
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
    disputes = [d for d in pipeline.disputes if (dispute_item_ids is None or d.dispute_item_id in dispute_item_ids)]

    question_by_key = {q.key: q for q in pipeline.questions}
    dispute_by_paper_version, candidates_by_paper_version = _dispute_index(disputes)
    all_papers = _papers_from_disputes(disputes)
    disputes_by_paper: dict[int, list[DisputeItem]] = {}
    for d in disputes:
        disputes_by_paper.setdefault(d.paper_id, []).append(d)

    n_saved = 0
    n_failed = 0

    for run in runs:
        cfg = effective_arbiter_config(run.config, settings.arbiter_run_defaults)
        _litellm.drop_params = cfg.drop_params
        groups_def: dict[int, list[list[Question]]] = {
            paper_id: resolve_groups(cfg, _questions_for_paper(paper_disputes, question_by_key))
            for paper_id, paper_disputes in disputes_by_paper.items()
        }

        source_texts: dict[int, str] = {}
        for paper in all_papers:
            if cfg.text_source == "candidates_only":
                source_texts[paper.paper_id] = ""
            elif cfg.text_source == "full_text":
                source = store.get_ocr(paper.paper_id)
                if source is None:
                    source = await client.fetch_ocr_markdown(paper.paper_id)
                    store.save_ocr(paper.paper_id, source)
                if source is not None:
                    source_texts[paper.paper_id] = source
                # papers without OCR are skipped silently for pass1
            else:
                source_texts[paper.paper_id] = paper.abstract

        pending_cells: dict[str, tuple] = {}
        # p1_errors unused here — this standalone pass1-only entrypoint already
        # infers failure from a missing cid in p1_texts; kept for the 4-tuple
        # return shape shared with run_arbitration_pipeline.
        p1_texts, p1_usage, p1_errors, p1_error = await _execute_pass1(
            run, cfg, all_papers, source_texts, pending_cells,
            store, settings, dry_run, groups_def=groups_def,
            sem=sem, limiter=limiter,
            build_p1_messages=_build_p1_messages_fn(cfg, candidates_by_paper_version),
            should_skip_cell=_skip_resolution_cell_fn(store),
        )
        if p1_error:
            logger.error("Pass-1 aborted for run %d: %s", run.run_id, p1_error)

        # Remove cells already pass1_done (should_skip_cell only checks done/posted)
        pass1_done_cids = set()
        for cid, (paper, group, group_idx) in list(pending_cells.items()):
            statuses = []
            for q in group:
                d = dispute_by_paper_version.get((paper.paper_id, q.version_id))
                if d is None:
                    continue
                statuses.append(store.get_resolution_status(run.run_id, d.dispute_item_id))
            if statuses and all(s == "pass1_done" for s in statuses):
                pass1_done_cids.add(cid)
        for cid in pass1_done_cids:
            del pending_cells[cid]

        for cid, (paper, group, group_idx) in pending_cells.items():
            p1_text = p1_texts.get(cid)
            if p1_text is None:
                n_failed += len(group)
                continue

            u = p1_usage.get(cid, {})
            tok_input = u.get("input_tokens", 0)
            tok_output = u.get("output_tokens", 0)
            tok_cached = u.get("cache_read_tokens", 0)
            p1_cost = u.get("cost")
            p1_latency = u.get("latency_ms", 0) or 0

            for i, question in enumerate(group):
                d = dispute_by_paper_version.get((paper.paper_id, question.version_id))
                if d is None:
                    continue

                raw_response: dict = {
                    "text_source": cfg.text_source,
                    "batch_group_id": cid,
                    "batch_mode": cfg.batch_p1,
                    "p1_usage": u,
                }
                if i == 0:
                    raw_response["pass1_text"] = p1_text

                payload = build_resolution(
                    arbiter_run_id=run.run_id,
                    paper_id=paper.paper_id,
                    dispute_item_id=d.dispute_item_id,
                    question=question,
                    value=None,
                    comment="",
                    cited_text="",
                    cited_text_verified=None,
                    raw_response=raw_response,
                    latency_ms=p1_latency if i == 0 else 0,
                    tokens_total=tok_input + tok_output + tok_cached if i == 0 else 0,
                    tokens_input=tok_input if i == 0 else 0,
                    tokens_output=tok_output if i == 0 else 0,
                    tokens_cached=tok_cached if i == 0 else 0,
                    cost=p1_cost if i == 0 else None,
                    cost_currency="USD",
                )
                store.save_pass1_resolution(
                    run.run_id, d.dispute_item_id, paper.paper_id, question.version_id, payload, cid
                )
                n_saved += 1

    logger.info("Arbitration Pass-1 complete: %d cells saved, %d failed", n_saved, n_failed)
    return n_saved, n_failed


async def arbitration_pass2_pipeline(
    pipeline: DisputePipelineConfig,
    settings: Settings,
    *,
    format_model: str | None = None,
    format_model_provider: str | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    dispute_item_ids: list[int] | None = None,
    concurrency: int | None = None,
    rpm: float | None = None,
    post: bool = True,
) -> tuple[int, int]:
    """Run only Pass-2 on pass1_done disputes, producing done resolutions.

    Returns (n_done, n_failed).
    """
    from .seer_client import DryRunSeerClient

    pipeline_cfg = effective_arbiter_config(ArbiterRunConfig(), settings.arbiter_run_defaults)
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
    disputes = [d for d in pipeline.disputes if (dispute_item_ids is None or d.dispute_item_id in dispute_item_ids)]

    question_map = {q.version_id: q for q in pipeline.questions}
    question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}
    dispute_by_paper_version, _ = _dispute_index(disputes)
    all_papers = _papers_from_disputes(disputes)

    n_done = 0
    n_failed = 0

    for run in runs:
        cfg = effective_arbiter_config(run.config, settings.arbiter_run_defaults)
        _litellm.drop_params = cfg.drop_params

        if format_model is not None:
            cfg.format_model = format_model
        if format_model_provider is not None:
            cfg.format_model_provider = format_model_provider

        for paper in all_papers:
            rows = store.get_pass1_resolution_rows(run.run_id, paper.paper_id)
            if not rows:
                continue

            source_text = _resolve_source_text(cfg, paper, store)

            groups: dict[str, list[dict]] = {}
            for row in rows:
                raw_resp = json.loads(row["payload"]["raw_response"])
                group_id = raw_resp.get("batch_group_id") or f"solo_{row['version_id']}"
                groups.setdefault(group_id, []).append(row)

            pending_cells: dict[str, tuple] = {}
            pending_p1: dict[str, str] = {}
            p1_usage_by_cid: dict[str, dict] = {}
            p1_payload_by_cid: dict[str, dict] = {}

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
                pass1_text = raw0.get("pass1_text")
                if not pass1_text:
                    logger.warning(
                        "No pass1_text in raw_response for run=%d paper=%d group=%s — skipping",
                        run.run_id, paper.paper_id, group_id,
                    )
                    n_failed += len(questions_in_group)
                    continue

                pending_cells[group_id] = (paper, questions_in_group, 0)
                pending_p1[group_id] = pass1_text
                p1_usage_by_cid[group_id] = raw0.get("p1_usage", {})
                p1_payload_by_cid[group_id] = ordered_rows[0]["payload"]

            if not pending_p1:
                continue

            # p2_errors unused here — this standalone pass2-only entrypoint
            # already infers failure from a missing cid in p2_texts; kept for the
            # 3-tuple return shape shared with run_arbitration_pipeline.
            p2_texts, p2_usage, p2_errors = await _execute_pass2(
                run, cfg, pending_p1, pending_cells,
                store, settings, dry_run,
                sem=sem, limiter=limiter,
            )

            for cid, p2_text in p2_texts.items():
                if cid not in pending_cells:
                    continue
                paper_cell, group, group_idx = pending_cells[cid]
                p1_payload_row0 = p1_payload_by_cid.get(cid, {})
                u = p1_usage_by_cid.get(cid, {})

                p1_tok_input = p1_payload_row0.get("tokens_input", 0) or 0
                p1_tok_output = p1_payload_row0.get("tokens_output", 0) or 0
                p1_tok_cached = p1_payload_row0.get("tokens_cached", 0) or 0
                p1_tok_total = p1_payload_row0.get("tokens_total", 0) or 0
                p1_cost_str = p1_payload_row0.get("cost")
                p1_cost = Decimal(p1_cost_str) if p1_cost_str else None
                p1_latency = p1_payload_row0.get("latency_ms", 0) or 0

                fu = p2_usage.get(cid, {})
                fmt_input = fu.get("input_tokens", 0)
                fmt_output = fu.get("output_tokens", 0)
                fmt_cached = fu.get("cache_read_tokens", 0)
                fmt_total = fmt_input + fmt_output + fmt_cached
                fmt_cost = fu.get("cost")
                p2_latency = fu.get("latency_ms", 0) or 0

                parsed = parse_structured_output(p2_text, [q.key for q in group])
                failed_keys = {r["key"]: r["parse_error"] for r in parsed if "parse_error" in r}
                if failed_keys:
                    err = ExtractionError(failed_keys)
                    logger.error(
                        "EXTRACTION FAILURE (pass2) — run %d, paper %d, group %s: %s",
                        run.run_id, paper_cell.paper_id, cid, err,
                    )
                    for q in group:
                        d = dispute_by_paper_version.get((paper_cell.paper_id, q.version_id))
                        if d is None:
                            continue
                        store.save_resolution(
                            run.run_id, d.dispute_item_id, paper_cell.paper_id, q.version_id,
                            build_error_resolution(
                                arbiter_run_id=run.run_id, paper_id=paper_cell.paper_id,
                                dispute_item_id=d.dispute_item_id, question=q, resolution_detail=str(err),
                            ),
                        )
                    n_failed += len(group)
                    continue

                for i, (question, result) in enumerate(zip(group, parsed)):
                    d = dispute_by_paper_version.get((paper_cell.paper_id, question.version_id))
                    if d is None:
                        continue

                    verify = verify_citation(
                        result.get("cited_text", ""),
                        source_text,
                        max_error_rate=cfg.citation_max_error_rate,
                        max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                    )
                    cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]

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

                    payload = build_resolution(
                        arbiter_run_id=run.run_id,
                        paper_id=paper_cell.paper_id,
                        dispute_item_id=d.dispute_item_id,
                        question=question,
                        value=result.get("value"),
                        comment=result.get("comment", ""),
                        cited_text=result.get("cited_text", ""),
                        cited_text_verified=cited_text_verified,
                        raw_response=raw_response,
                        latency_ms=(p1_latency + p2_latency) if i == 0 else 0,
                        tokens_total=p1_tok_total if i == 0 else 0,
                        tokens_input=p1_tok_input if i == 0 else 0,
                        tokens_output=p1_tok_output if i == 0 else 0,
                        tokens_cached=p1_tok_cached if i == 0 else 0,
                        cost=p1_cost if i == 0 else None,
                        cost_currency="USD",
                        fmt_tokens_total=fmt_total if i == 0 else 0,
                        fmt_tokens_input=fmt_input if i == 0 else 0,
                        fmt_tokens_output=fmt_output if i == 0 else 0,
                        fmt_tokens_cached=fmt_cached if i == 0 else 0,
                        fmt_cost=fmt_cost if i == 0 else None,
                        confidence=result.get("confidence"),
                    )
                    store.save_resolution(
                        run.run_id, d.dispute_item_id, paper_cell.paper_id, question.version_id, payload, cid
                    )
                    n_done += 1

            if post:
                unposted = store.get_unposted_resolutions(run.run_id, paper.paper_id)
                if unposted:
                    try:
                        await client.post_resolutions_bulk(unposted)
                        if not dry_run:
                            dispute_ids = [r["dispute_item"] for r in unposted]
                            store.mark_resolutions_posted(run.run_id, dispute_ids)
                        logger.info("Posted %d resolutions for paper %d", len(unposted), paper.paper_id)
                    except Exception as exc:
                        logger.error("Post failed for paper %d: %s", paper.paper_id, exc)

    logger.info("Arbitration Pass-2 complete: %d done, %d failed", n_done, n_failed)
    return n_done, n_failed


async def reformat_arbitration_pipeline(
    pipeline: DisputePipelineConfig,
    settings: Settings,
    *,
    format_model: str | None = None,
    format_model_provider: str | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    dispute_item_ids: list[int] | None = None,
) -> tuple[int, int]:
    """Re-run pass-2 formatting on stored resolutions, optionally with a different model.

    Returns (n_updated, n_failed). Reuses annotate.engine.reformat_group() unchanged —
    Pass-2-only reformatting needs no candidates, only the stored pass1_text.
    """
    import functools

    from .annotate.engine import reformat_group
    from .llm import complete as llm_complete, dummy_complete

    pipeline_cfg = effective_arbiter_config(ArbiterRunConfig(), settings.arbiter_run_defaults)

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params
    _litellm.suppress_debug_info = True

    store = Store(settings.runtime.store_path)
    disputes = [d for d in pipeline.disputes if (dispute_item_ids is None or d.dispute_item_id in dispute_item_ids)]
    question_map = {q.version_id: q for q in pipeline.questions}
    question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}
    all_papers = _papers_from_disputes(disputes)
    paper_map = {p.paper_id: p for p in all_papers}

    _base_complete = dummy_complete if dry_run else llm_complete
    _limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    async def _rate_limited(model, provider, messages, **kw):
        await _limiter.acquire(provider)
        return await _base_complete(model, provider, messages, **kw)

    sem = asyncio.Semaphore(pipeline_cfg.concurrency)

    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]

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
        run_task = progress.add_task("Arbiter runs", total=len(runs))
        paper_task = progress.add_task("", total=len(all_papers), visible=False)

        for run in runs:
            progress.update(run_task, description=f"Run {run.run_id} [cyan]({run.name})[/cyan]")
            progress.reset(paper_task, total=len(all_papers), visible=True)

            cfg = effective_arbiter_config(run.config, settings.arbiter_run_defaults)
            _litellm.drop_params = cfg.drop_params
            final_model = format_model or cfg.format_model or "gpt-4o-mini"
            final_provider = format_model_provider or cfg.format_model_provider or run.model_provider

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

            for paper in all_papers:
                progress.update(paper_task, description=f"  Paper {paper.paper_id}")

                rows = store.get_reformattable_resolution_rows(run.run_id, paper.paper_id)
                if not rows:
                    progress.advance(paper_task)
                    continue

                source_text = _resolve_source_text(cfg, paper_map.get(paper.paper_id, paper), store)

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
                        new_payload = build_resolution(
                            arbiter_run_id=row["arbiter_run_id"],
                            paper_id=row["paper_id"],
                            dispute_item_id=row["dispute_item_id"],
                            question=question,
                            value=parsed.get("value"),
                            comment=parsed.get("comment", ""),
                            cited_text=parsed.get("cited_text", ""),
                            cited_text_verified=result["cited_text_verified"],
                            raw_response=new_raw,
                            latency_ms=old_payload.get("latency_ms", 0),
                            tokens_total=old_payload.get("tokens_total", 0),
                            tokens_input=old_payload.get("tokens_input", 0),
                            tokens_output=old_payload.get("tokens_output", 0),
                            tokens_cached=old_payload.get("tokens_cached", 0),
                            cost=Decimal(old_payload["cost"]) if old_payload.get("cost") else None,
                            cost_currency=old_payload.get("cost_currency", "USD"),
                            fmt_tokens_total=result["fmt_tokens_total"],
                            fmt_tokens_input=result["fmt_tokens_input"],
                            fmt_tokens_output=result["fmt_tokens_output"],
                            fmt_tokens_cached=result["fmt_tokens_cached"],
                            fmt_cost=result["fmt_cost"],
                            confidence=parsed.get("confidence"),
                        )
                        store.update_reformatted_resolution(
                            row["arbiter_run_id"], row["dispute_item_id"], new_payload
                        )
                        total_done += 1

                await asyncio.gather(*[_do_group(gid, grows) for gid, grows in groups.items()])

                progress.advance(paper_task)

            progress.advance(run_task)

    logger.info("Arbitration reformat complete: %d updated, %d failed", total_done, total_failed)
    return total_done, total_failed
