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

from .config import PipelineConfig, Settings
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
def run(
    pipeline_json: str,
    settings_path: str | None,
    runs: str | None,
    papers: str | None,
    dry_run: bool,
) -> None:
    """Execute annotation for PIPELINE_JSON."""
    from .orchestrator import run_pipeline

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    log_path = pathlib.Path("seer-annotate.log")
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
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed")
        console.print(f"\n[bold red]Error:[/] {exc}")
        console.print("[dim]Full traceback written to seer-annotate.log[/]")
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
