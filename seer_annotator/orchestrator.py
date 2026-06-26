"""Main annotation loop: run → paper → group → annotate → persist → post."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from decimal import Decimal
from typing import Callable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .batching import resolve_groups
from .batch_runner import run_batch_pipeline
from .config import PipelineConfig, ProviderSettings, RunConfig, Settings, effective_run_config
from .rate_limiter import PerProviderRateLimiter
from .seer_client import SeerClient
from .store import Store
from .llm import complete as llm_complete, dummy_complete
from .annotate.engine import annotate_group, CompleteFn
from .mapping import build_error_answer

logger = logging.getLogger(__name__)


async def run_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
    store: Store | None = None,
    client: SeerClient | None = None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
) -> None:
    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    effective = {r.run_id: effective_run_config(r.config, settings.run_defaults) for r in runs}

    if any(effective[r.run_id].batch_p1 or effective[r.run_id].batch_p2 for r in runs):
        return await run_batch_pipeline(
            pipeline,
            settings,
            store=store,
            client=client,
            dry_run=dry_run,
            run_ids=run_ids,
            paper_ids=paper_ids,
        )

    from .seer_client import DryRunSeerClient

    # Pipeline-wide settings come from run_defaults merged over code defaults.
    # Per-run overrides for these (concurrency, rpm, drop_params) are not supported
    # because the semaphore and rate limiter are created once for the whole pipeline.
    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params

    store = store or Store(settings.runtime.store_path)
    client = client or (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    _base_complete = dummy_complete if dry_run else llm_complete
    _limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    async def _rate_limited_complete(model, provider, messages, **kw):
        await _limiter.acquire(provider)
        return await _base_complete(model, provider, messages, **kw)

    complete_fn = _rate_limited_complete
    sem = asyncio.Semaphore(pipeline_cfg.concurrency)
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]


    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

    first_error: list[Exception] = []  # mutable; set once on first LLM failure

    with progress:
        run_task = progress.add_task("Runs", total=len(runs))
        paper_task = progress.add_task("", total=len(papers), visible=False)

        for run in runs:
            progress.update(
                run_task,
                description=f"Run {run.run_id} [cyan]({run.name})[/cyan]",
            )
            progress.reset(paper_task, total=len(papers), visible=True)
            progress.update(paper_task, description=f"  Papers")
            logger.debug("Run %d (%s) starting", run.run_id, run.name)

            cfg = effective[run.run_id]

            # Inject provider-specific credentials if configured
            run_complete_fn = complete_fn
            if not dry_run:
                prov = settings.providers.get(run.model_provider, ProviderSettings())
                extra: dict = {}
                api_key = prov.resolved_api_key()
                if api_key:
                    extra["api_key"] = api_key
                if prov.base_url:
                    extra["api_base"] = prov.base_url
                if prov.api_version:
                    extra["api_version"] = prov.api_version
                if extra:
                    run_complete_fn = functools.partial(complete_fn, **extra)

            effective_fmt_provider = cfg.format_model_provider or run.model_provider
            fmt_complete_fn: CompleteFn | None = None
            if not dry_run and effective_fmt_provider != run.model_provider:
                fmt_prov = settings.providers.get(effective_fmt_provider, ProviderSettings())
                fmt_extra: dict = {}
                fmt_api_key = fmt_prov.resolved_api_key()
                if fmt_api_key:
                    fmt_extra["api_key"] = fmt_api_key
                if fmt_prov.base_url:
                    fmt_extra["api_base"] = fmt_prov.base_url
                if fmt_prov.api_version:
                    fmt_extra["api_version"] = fmt_prov.api_version
                fmt_complete_fn = functools.partial(complete_fn, **fmt_extra) if fmt_extra else complete_fn

            groups_def = resolve_groups(cfg, pipeline.questions)

            for paper in papers:
                progress.update(
                    paper_task,
                    description=f"  Paper {paper.paper_id}",
                )
                logger.debug("Paper %d", paper.paper_id)

                # Resolve source text
                if cfg.text_source == "full_text":
                    source = store.get_ocr(paper.paper_id)
                    if source is None:
                        source = await client.fetch_ocr_markdown(paper.paper_id)
                        store.save_ocr(paper.paper_id, source)
                    if source is None:
                        logger.warning("No OCR for paper %d — posting error for all questions", paper.paper_id)
                        for q in pipeline.questions:
                            store.save_answer(
                                run.run_id, paper.paper_id, q.version_id,
                                build_error_answer(run_id=run.run_id, paper_id=paper.paper_id, question=q, extraction_detail="no_ocr"),
                            )
                        unposted = store.get_unposted(run.run_id, paper.paper_id)
                        if unposted:
                            try:
                                await client.post_answers_bulk(unposted)
                                if not dry_run:
                                    store.mark_posted(run.run_id, paper.paper_id, [p["question_version"] for p in unposted])
                            except Exception as exc:
                                logger.error("Post failed for paper %d: %s", paper.paper_id, exc)
                        progress.advance(paper_task)
                        continue
                else:
                    source = paper.abstract

                # Process groups
                async def _do_group(group: list, group_idx: int) -> None:
                    if first_error:
                        return
                    cells = [(run.run_id, paper.paper_id, q.version_id) for q in group]
                    if all(store.should_skip_cell(*c) for c in cells):
                        logger.debug("Group %d already done, skipping", group_idx)
                        return

                    async with sem:
                        try:
                            payloads = await annotate_group(
                                run=run,
                                paper_id=paper.paper_id,
                                source_text=source,
                                questions=group,
                                format_model=cfg.format_model or "gpt-4o-mini",
                                format_model_provider=effective_fmt_provider,
                                complete_fn=run_complete_fn,
                                format_complete_fn=fmt_complete_fn,
                                system_prompt=cfg.system_prompt,
                                citation_max_error_rate=cfg.citation_max_error_rate,
                                citation_max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                            )
                        except Exception as exc:
                            logger.error("Group %d failed: %s", group_idx, exc)
                            if cfg.fail_fast and not first_error:
                                first_error.append(exc)
                            for q in group:
                                store.save_answer(
                                    run.run_id, paper.paper_id, q.version_id,
                                    build_error_answer(run_id=run.run_id, paper_id=paper.paper_id, question=q, extraction_detail=str(exc)),
                                )
                            return

                    for payload, (_, _, vid) in zip(payloads, cells):
                        store.save_answer(run.run_id, paper.paper_id, vid, payload)

                await asyncio.gather(*[_do_group(g, i) for i, g in enumerate(groups_def)])

                if first_error:
                    break

                # Post all done answers for this paper
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

                progress.advance(paper_task)

            if first_error:
                break
            progress.advance(run_task)
            logger.debug("Run %d complete", run.run_id)

    if first_error:
        raise first_error[0]


async def reformat_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    *,
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
    from .annotate.engine import reformat_group
    from .annotate.parse import ExtractionError
    from .mapping import build_llm_answer

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params

    store = Store(settings.runtime.store_path)
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
                    source_text = store.get_ocr(paper.paper_id) or ""
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
                        new_payload = build_llm_answer(
                            run_id=row["run_id"],
                            paper_id=row["paper_id"],
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
                        store.update_reformatted(row["run_id"], row["paper_id"], row["version_id"], new_payload)
                        total_done += 1

                await asyncio.gather(*[_do_group(gid, grows) for gid, grows in groups.items()])

                progress.advance(paper_task)

            progress.advance(run_task)

    logger.info("Reformat complete: %d updated, %d failed", total_done, total_failed)
    return total_done, total_failed
