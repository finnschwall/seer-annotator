"""CLI: seer-annotate run / status."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sys

import click
from rich.console import Console
from rich.table import Table

from .config import DisputePipelineConfig, PipelineConfig, Settings
from .orchestrator import pass1_pipeline, pass2_pipeline
from .store import Store

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def cli(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        for noisy in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to process")
@click.option("--papers", default=None, help="Comma-separated paper IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--chunk-papers", default=None, type=int, help="Override chunk_papers for all runs")
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
@click.option(
    "--progress-url", default=None,
    help="POST heartbeats here (run start / after each chunk / on completion) — used by SEER-orchestrated runs",
)
@click.option("--log-file", "log_file", default="seer-annotate.log", help="Path to write the run log to")
@click.option(
    "--reset-runs", default=None,
    help="Comma-separated run IDs whose cached answers should be cleared from the local store "
    "before running, forcing fresh re-annotation instead of skipping cells the resume cache "
    "already thinks are done/posted. Other run IDs sharing the same store are left untouched.",
)
def run(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
    chunk_papers: int | None,
    concurrency: int | None,
    rpm: float | None,
    progress_url: str | None,
    log_file: str,
    reset_runs: str | None,
) -> None:
    """Execute annotation for PIPELINE_JSON."""
    from .orchestrator import run_pipeline
    from .progress import ProgressReporter

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    if reset_runs:
        reset_ids = [int(x) for x in reset_runs.split(",")]
        Store(settings.runtime.store_path).reset_runs(reset_ids)
        console.print(f"[yellow]Cleared cached answers for run(s) {reset_ids}[/]")

    if chunk_papers is not None:
        for r in pipeline.runs:
            r.config.chunk_papers = chunk_papers

    log_path = pathlib.Path(log_file)
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    console.print(f"[dim]Logging to {log_path}[/]")

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider, posts are printed[/]")

    try:
        asyncio.run(
            run_pipeline(
                pipeline,
                settings,
                dry_run=dry_run,
                run_ids=run_ids,
                paper_ids=paper_ids,
                concurrency=concurrency,
                rpm=rpm,
                progress_url=progress_url,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        console.print(f"[dim]Full traceback written to {log_path}[/]")
        if progress_url and pipeline.runs:
            reporter = ProgressReporter(progress_url, pipeline.api_token, pipeline.runs[0].run_id)
            asyncio.run(
                reporter.heartbeat(
                    status="failed", cells_total=0, cells_done=0, cells_error=0, message=str(exc),
                )
            )
        sys.exit(1)
    console.print("[green]Done.[/]")


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
def status(pipeline_json: str, settings_path: str | None) -> None:
    """Show local store progress per run."""
    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)
    store = Store(settings.runtime.store_path)

    overall = store.stats()
    console.print(f"[bold]OCR cached:[/] {overall['ocr_cached']}")
    console.print(f"[bold]Answer counts by status:[/] {overall['answers']}")

    cost_rows = store.cost_summary()
    if cost_rows:
        run_map = {r.run_id: r.name for r in pipeline.runs}
        table = Table(title="Token / Cost Summary")
        table.add_column("Run", style="cyan")
        table.add_column("Answers", justify="right")
        table.add_column("Tokens (total)", justify="right")
        table.add_column("Cached", justify="right")
        table.add_column("Cost (USD)", justify="right")
        for row in cost_rows:
            table.add_row(
                run_map.get(row["run_id"], str(row["run_id"])),
                str(row["answers"]),
                f"{row['tokens_total']:,}",
                f"{row['tokens_cached']:,}",
                f"${row['cost_usd']:.4f}",
            )
        console.print(table)


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to repost")
@click.option("--papers", default=None, help="Comma-separated paper IDs to repost")
@click.option("--dry-run", is_flag=True)
def repost(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
) -> None:
    """Re-post stored answers to the server (e.g. after a DB reset)."""
    from .seer_client import SeerClient, DryRunSeerClient

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)
    store = Store(settings.runtime.store_path)
    client = (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    runs_to_process = [r for r in pipeline.runs if run_ids is None or r.run_id in run_ids]
    papers_to_process = [p for p in pipeline.papers if paper_ids is None or p.paper_id in paper_ids]

    if dry_run:
        console.print("[yellow]Dry-run mode — posts are printed[/]")

    total = 0

    async def _do() -> None:
        nonlocal total
        for run in runs_to_process:
            for paper in papers_to_process:
                answers = store.get_postable(run.run_id, paper.paper_id)
                if not answers:
                    continue
                await client.post_answers_bulk(answers)
                if not dry_run:
                    version_ids = [a["question_version"] for a in answers]
                    store.mark_posted(run.run_id, paper.paper_id, version_ids)
                total += len(answers)

    try:
        asyncio.run(_do())
    except Exception as exc:
        logging.getLogger(__name__).exception("Repost failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[green]Re-posted {total} answers.[/]")


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--format-model", default=None, help="Override the format model (e.g. gpt-4o-mini)")
@click.option("--format-model-provider", default=None, help="Override the format model provider")
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to reformat")
@click.option("--papers", default=None, help="Comma-separated paper IDs to reformat")
@click.option("--dry-run", is_flag=True)
def reformat(
    pipeline_json: str,
    format_model: str | None,
    format_model_provider: str | None,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
) -> None:
    """Re-run pass-2 formatting on stored answers without re-running the reasoning model.

    Useful for upgrading the format model or comparing different format models.
    After reformatting, answers are reset to status 'done' and can be reposted
    with the repost command.
    """
    from .orchestrator import reformat_pipeline

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    effective_model = format_model or settings.run_defaults.format_model
    console.print(f"[dim]Format model: {effective_model}[/]")

    try:
        n_done, n_failed = asyncio.run(
            reformat_pipeline(
                pipeline,
                settings,
                format_model=format_model,
                format_model_provider=format_model_provider,
                dry_run=dry_run,
                run_ids=run_ids,
                paper_ids=paper_ids,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Reformat failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Reformat done: {n_done} updated, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Reformat done: {n_done} answers updated.[/]")
    if n_done and not dry_run:
        console.print("[dim]Run 'seer-annotate repost' to push the updated answers to SEER.[/]")


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to process")
@click.option("--papers", default=None, help="Comma-separated paper IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
def pass1(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
    concurrency: int | None,
    rpm: float | None,
) -> None:
    """Run only Pass-1 (reasoning) and store results as pass1_done for a later pass2.

    Executes the expensive reasoning model for each pending cell and persists the
    raw reasoning text with status 'pass1_done'. No answers are posted to SEER.
    Run 'seer-annotate pass2' afterwards to format and post results.
    """
    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    try:
        n_done, n_failed = asyncio.run(
            pass1_pipeline(
                pipeline,
                settings,
                dry_run=dry_run,
                run_ids=run_ids,
                paper_ids=paper_ids,
                concurrency=concurrency,
                rpm=rpm,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pass 1 failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Pass 1 complete: {n_done} cells, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Pass 1 complete: {n_done} cells, {n_failed} failed.[/]")


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--format-model", default=None, help="Override the format model (e.g. gpt-4o-mini)")
@click.option("--format-model-provider", default=None, help="Override the format model provider")
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to process")
@click.option("--papers", default=None, help="Comma-separated paper IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
@click.option("--no-post", is_flag=True, help="Skip posting answers to SEER (leave as 'done')")
def pass2(
    pipeline_json: str,
    format_model: str | None,
    format_model_provider: str | None,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
    concurrency: int | None,
    rpm: float | None,
    no_post: bool,
) -> None:
    """Pick up pass1_done cells, run Pass-2 (formatting), and post answers to SEER.

    Reads cells with status 'pass1_done' (written by 'seer-annotate pass1'), runs the
    cheap format model to produce typed JSON answers, and posts them to SEER unless
    --no-post is given. Use --no-post followed by 'seer-annotate repost' to post later.
    """
    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    effective_model = format_model or settings.run_defaults.format_model
    console.print(f"[dim]Format model: {effective_model}[/]")

    try:
        n_done, n_failed = asyncio.run(
            pass2_pipeline(
                pipeline,
                settings,
                format_model=format_model,
                format_model_provider=format_model_provider,
                dry_run=dry_run,
                run_ids=run_ids,
                paper_ids=paper_ids,
                concurrency=concurrency,
                rpm=rpm,
                post=not no_post,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pass 2 failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Pass 2 complete: {n_done} cells, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Pass 2 complete: {n_done} cells, {n_failed} failed.[/]")
    if n_done and not dry_run and no_post:
        console.print("[dim]Run 'seer-annotate repost' to push the answers to SEER.[/]")


@cli.command(name="preview-prompt")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated run IDs to include")
@click.option("--papers", default=None, help="Comma-separated paper IDs to include")
@click.option(
    "--pass", "which_pass",
    default="both",
    type=click.Choice(["1", "2", "both"]),
    help="Which pass to show (default: both)",
)
@click.option("--output", "output_path", default=None, help="Write to file instead of stdout")
@click.option(
    "--no-fetch",
    is_flag=True,
    help="Do not fetch OCR from SEER; use store cache only (placeholder text if not cached)",
)
def preview_prompt(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    which_pass: str,
    output_path: str | None,
    no_fetch: bool,
) -> None:
    """Show the exact LLM prompt(s) for each (run, paper, question-group) cell.

    Reflects the real batching, caching markers, system prompt, and pass-2 format
    instructions. Pass-2 is shown with a placeholder for the pass-1 output since no
    actual inference is done.

    OCR text is read from the local store cache. Use --no-fetch to suppress the
    SEER network call when OCR is not yet cached (a placeholder is used instead).
    """
    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = {int(x) for x in runs.split(",")} if runs else None
    paper_ids = {int(x) for x in papers.split(",")} if papers else None

    runs_to_process = [r for r in pipeline.runs if run_ids is None or r.run_id in run_ids]
    papers_to_process = [p for p in pipeline.papers if paper_ids is None or p.paper_id in paper_ids]

    if not runs_to_process:
        console.print("[yellow]No matching runs.[/]")
        return
    if not papers_to_process:
        console.print("[yellow]No matching papers.[/]")
        return

    from .annotate.prompt import build_messages, build_format_messages
    from .batching import resolve_groups
    from .caching import apply_cache
    from .config import effective_run_config
    from .store import Store
    from .seer_client import SeerClient

    store = Store(settings.runtime.store_path)

    async def _fetch_ocr(paper_id: int) -> str | None:
        cached = store.get_ocr(paper_id)
        if cached is not None:
            return cached
        if no_fetch:
            return None
        client = SeerClient(pipeline.api_base, pipeline.api_token)
        text = await client.fetch_ocr_markdown(paper_id)
        if text is not None:
            store.save_ocr(paper_id, text)
        return text

    lines: list[str] = []

    async def _build() -> None:
        for exp_run in runs_to_process:
            cfg = effective_run_config(exp_run.config, settings.run_defaults)
            groups = resolve_groups(cfg, pipeline.questions)

            for paper in papers_to_process:
                ocr_text = await _fetch_ocr(paper.paper_id)
                if ocr_text is None:
                    ocr_text = "[OCR TEXT NOT AVAILABLE — run the pipeline first or remove --no-fetch]"

                source_text = paper.abstract if cfg.text_source == "abstract" else ocr_text

                for group_idx, question_group in enumerate(groups):
                    keys = ", ".join(q.key for q in question_group)
                    header = (
                        f"{'=' * 80}\n"
                        f"RUN:    {exp_run.name}  (id={exp_run.run_id})\n"
                        f"MODEL:  {exp_run.model_provider}/{exp_run.model_name}\n"
                        f"PAPER:  {paper.title}  (id={paper.paper_id})\n"
                        f"GROUP:  {group_idx + 1}/{len(groups)}  keys=[{keys}]\n"
                        f"CACHE:  {'on' if cfg.cache else 'off'}"
                        f"  batching={cfg.batching}  text_source={cfg.text_source}\n"
                        f"{'=' * 80}"
                    )
                    lines.append(header)

                    if which_pass in ("1", "both"):
                        msgs = build_messages(
                            source_text,
                            question_group,
                            text_source=cfg.text_source,
                            system_prompt=cfg.system_prompt,
                            cache_first=cfg.cache_first,
                        )
                        msgs = apply_cache(
                            exp_run.model_provider,
                            msgs,
                            enabled=cfg.cache,
                            ttl=cfg.cache_ttl,
                        )
                        lines.append("\n--- PASS 1 ---\n")
                        for msg in msgs:
                            role = msg["role"].upper()
                            content = msg["content"]
                            if isinstance(content, list):
                                # cache-annotated content blocks
                                text_parts = []
                                for block in content:
                                    if isinstance(block, dict):
                                        text_parts.append(block.get("text", repr(block)))
                                    else:
                                        text_parts.append(str(block))
                                content_str = "".join(text_parts)
                            else:
                                content_str = content
                            lines.append(f"[{role}]\n{content_str}\n")

                    if which_pass in ("2", "both"):
                        placeholder = (
                            "[PASS-1 OUTPUT WOULD GO HERE — "
                            "this is the reasoning text that pass-2 reformats into JSON]"
                        )
                        fmt_msgs = build_format_messages(placeholder, question_group)
                        lines.append("\n--- PASS 2 ---\n")
                        for msg in fmt_msgs:
                            role = msg["role"].upper()
                            lines.append(f"[{role}]\n{msg['content']}\n")

    asyncio.run(_build())

    output = "\n".join(lines)

    if output_path:
        pathlib.Path(output_path).write_text(output, encoding="utf-8")
        console.print(f"[green]Wrote {len(lines)} sections to {output_path}[/]")
    else:
        click.echo(output)


@cli.command(name="ui")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8765, type=int)
def ui_cmd(pipeline_json: str, settings_path: str | None, host: str, port: int) -> None:
    """Launch the debug web UI."""
    import uvicorn
    from .ui.app import create_app

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)
    app = create_app(pipeline=pipeline, settings=settings)

    console.print(f"[green]UI:[/] http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Arbitration commands — mirror the annotation commands above, operating on a
# dispute-set pipeline JSON (DisputePipelineConfig) instead of PipelineConfig.
# ---------------------------------------------------------------------------

def _load_dispute_pipeline(pipeline_json: str) -> DisputePipelineConfig:
    with open(pipeline_json) as f:
        return DisputePipelineConfig.model_validate(json.load(f))


@cli.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated arbiter run IDs to process")
@click.option("--disputes", default=None, help="Comma-separated dispute_item IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--chunk-papers", default=None, type=int, help="Override chunk_papers for all runs")
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
def arbitrate(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    disputes: str | None,
    dry_run: bool,
    chunk_papers: int | None,
    concurrency: int | None,
    rpm: float | None,
) -> None:
    """Adjudicate disputes for DISPUTE_PIPELINE_JSON (Pass-1 + Pass-2 + post, chunked)."""
    from .arbitrate_orchestrator import run_arbitration_pipeline

    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    dispute_item_ids = [int(x) for x in disputes.split(",")] if disputes else None

    if chunk_papers is not None:
        for r in pipeline.runs:
            r.config.chunk_papers = chunk_papers

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider, posts are printed[/]")

    try:
        asyncio.run(
            run_arbitration_pipeline(
                pipeline,
                settings,
                dry_run=dry_run,
                run_ids=run_ids,
                dispute_item_ids=dispute_item_ids,
                concurrency=concurrency,
                rpm=rpm,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Arbitration pipeline failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)
    console.print("[green]Done.[/]")


@cli.command(name="arbitrate-status")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
def arbitrate_status(pipeline_json: str, settings_path: str | None) -> None:
    """Show local store progress for a dispute-set pipeline."""
    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)
    store = Store(settings.runtime.store_path)

    console.print(f"[bold]Resolution counts by status:[/] {store.resolution_stats()}")

    cost_rows = store.resolution_cost_summary()
    if cost_rows:
        run_map = {r.run_id: r.name for r in pipeline.runs}
        table = Table(title="Token / Cost Summary")
        table.add_column("Run", style="cyan")
        table.add_column("Resolutions", justify="right")
        table.add_column("Tokens (total)", justify="right")
        table.add_column("Cached", justify="right")
        table.add_column("Cost (USD)", justify="right")
        for row in cost_rows:
            table.add_row(
                run_map.get(row["run_id"], str(row["run_id"])),
                str(row["answers"]),
                f"{row['tokens_total']:,}",
                f"{row['tokens_cached']:,}",
                f"${row['cost_usd']:.4f}",
            )
        console.print(table)


@cli.command(name="arbitrate-repost")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated arbiter run IDs to repost")
@click.option("--dry-run", is_flag=True)
def arbitrate_repost(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    dry_run: bool,
) -> None:
    """Re-post stored resolutions to the server (e.g. after a DB reset)."""
    from .seer_client import SeerClient, DryRunSeerClient
    from .arbitrate_orchestrator import _papers_from_disputes

    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)
    store = Store(settings.runtime.store_path)
    client = (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    runs_to_process = [r for r in pipeline.runs if run_ids is None or r.run_id in run_ids]
    all_papers = _papers_from_disputes(pipeline.disputes)

    if dry_run:
        console.print("[yellow]Dry-run mode — posts are printed[/]")

    total = 0

    async def _do() -> None:
        nonlocal total
        for run in runs_to_process:
            for paper in all_papers:
                resolutions = store.get_postable_resolutions(run.run_id, paper.paper_id)
                if not resolutions:
                    continue
                await client.post_resolutions_bulk(resolutions)
                if not dry_run:
                    dispute_ids = [r["dispute_item"] for r in resolutions]
                    store.mark_resolutions_posted(run.run_id, dispute_ids)
                total += len(resolutions)

    try:
        asyncio.run(_do())
    except Exception as exc:
        logging.getLogger(__name__).exception("Repost failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[green]Re-posted {total} resolutions.[/]")


@cli.command(name="arbitrate-reformat")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--format-model", default=None, help="Override the format model (e.g. gpt-4o-mini)")
@click.option("--format-model-provider", default=None, help="Override the format model provider")
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated arbiter run IDs to reformat")
@click.option("--disputes", default=None, help="Comma-separated dispute_item IDs to reformat")
@click.option("--dry-run", is_flag=True)
def arbitrate_reformat(
    pipeline_json: str,
    format_model: str | None,
    format_model_provider: str | None,
    settings_path: str | None,
    runs: str | None,
    disputes: str | None,
    dry_run: bool,
) -> None:
    """Re-run pass-2 formatting on stored resolutions without re-adjudicating.

    After reformatting, resolutions are reset to status 'done' and can be reposted
    with arbitrate-repost.
    """
    from .arbitrate_orchestrator import reformat_arbitration_pipeline

    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    dispute_item_ids = [int(x) for x in disputes.split(",")] if disputes else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    effective_model = format_model or settings.arbiter_run_defaults.format_model
    console.print(f"[dim]Format model: {effective_model}[/]")

    try:
        n_done, n_failed = asyncio.run(
            reformat_arbitration_pipeline(
                pipeline,
                settings,
                format_model=format_model,
                format_model_provider=format_model_provider,
                dry_run=dry_run,
                run_ids=run_ids,
                dispute_item_ids=dispute_item_ids,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Arbitration reformat failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Reformat done: {n_done} updated, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Reformat done: {n_done} resolutions updated.[/]")
    if n_done and not dry_run:
        console.print("[dim]Run 'seer-annotate arbitrate-repost' to push the updated resolutions to SEER.[/]")


@cli.command(name="arbitrate-pass1")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated arbiter run IDs to process")
@click.option("--disputes", default=None, help="Comma-separated dispute_item IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
def arbitrate_pass1(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    disputes: str | None,
    dry_run: bool,
    concurrency: int | None,
    rpm: float | None,
) -> None:
    """Run only Pass-1 (adjudication reasoning) and store results as pass1_done.

    Run 'seer-annotate arbitrate-pass2' afterwards to format and post results.
    """
    from .arbitrate_orchestrator import arbitration_pass1_pipeline

    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    dispute_item_ids = [int(x) for x in disputes.split(",")] if disputes else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    try:
        n_done, n_failed = asyncio.run(
            arbitration_pass1_pipeline(
                pipeline,
                settings,
                dry_run=dry_run,
                run_ids=run_ids,
                dispute_item_ids=dispute_item_ids,
                concurrency=concurrency,
                rpm=rpm,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Arbitration Pass 1 failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Pass 1 complete: {n_done} cells, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Pass 1 complete: {n_done} cells, {n_failed} failed.[/]")


@cli.command(name="arbitrate-pass2")
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option("--format-model", default=None, help="Override the format model (e.g. gpt-4o-mini)")
@click.option("--format-model-provider", default=None, help="Override the format model provider")
@click.option("--settings", "settings_path", default=None)
@click.option("--runs", default=None, help="Comma-separated arbiter run IDs to process")
@click.option("--disputes", default=None, help="Comma-separated dispute_item IDs to process")
@click.option("--dry-run", is_flag=True)
@click.option("--concurrency", default=None, type=int, help="Override max concurrency")
@click.option("--rpm", default=None, type=float, help="Override requests-per-minute limit")
@click.option("--no-post", is_flag=True, help="Skip posting resolutions to SEER (leave as 'done')")
def arbitrate_pass2(
    pipeline_json: str,
    format_model: str | None,
    format_model_provider: str | None,
    settings_path: str | None,
    runs: str | None,
    disputes: str | None,
    dry_run: bool,
    concurrency: int | None,
    rpm: float | None,
    no_post: bool,
) -> None:
    """Pick up pass1_done disputes, run Pass-2 (formatting), and post resolutions.

    Use --no-post followed by 'seer-annotate arbitrate-repost' to post later.
    """
    from .arbitrate_orchestrator import arbitration_pass2_pipeline

    pipeline = _load_dispute_pipeline(pipeline_json)
    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    dispute_item_ids = [int(x) for x in disputes.split(",")] if disputes else None

    if dry_run:
        console.print("[yellow]Dry-run mode — LLM calls use dummy provider[/]")

    effective_model = format_model or settings.arbiter_run_defaults.format_model
    console.print(f"[dim]Format model: {effective_model}[/]")

    try:
        n_done, n_failed = asyncio.run(
            arbitration_pass2_pipeline(
                pipeline,
                settings,
                format_model=format_model,
                format_model_provider=format_model_provider,
                dry_run=dry_run,
                run_ids=run_ids,
                dispute_item_ids=dispute_item_ids,
                concurrency=concurrency,
                rpm=rpm,
                post=not no_post,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Arbitration Pass 2 failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        sys.exit(1)

    if n_failed:
        console.print(f"[yellow]Pass 2 complete: {n_done} cells, {n_failed} failed (see logs)[/]")
    else:
        console.print(f"[green]Pass 2 complete: {n_done} cells, {n_failed} failed.[/]")
    if n_done and not dry_run and no_post:
        console.print("[dim]Run 'seer-annotate arbitrate-repost' to push the resolutions to SEER.[/]")
