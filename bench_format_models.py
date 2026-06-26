#!/usr/bin/env python3
"""Benchmark multiple Pass-2 formatting models against stored Pass-1 outputs.

Results are written to a separate ``format_bench`` table so the main ``answers``
table is never touched.  Re-running the same model overwrites its previous rows.

Usage
-----
    python bench_format_models.py pipeline.json \\
        --model openai/gpt-4o-mini \\
        --model anthropic/claude-haiku-4-5-20251001 \\
        --model my_server/Qwen3-8B      # provider key from settings.toml

Provider credentials (api_key, base_url, api_version) are read from settings.toml
exactly as the main CLI does — no extra flags needed.
"""

from __future__ import annotations

import asyncio
import difflib
import functools
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from seer_annotator.annotate.engine import reformat_group
from seer_annotator.annotate.parse import ExtractionError
from seer_annotator.config import (
    PipelineConfig,
    ProviderSettings,
    RunConfig,
    Settings,
    effective_run_config,
)
from seer_annotator.llm import complete as llm_complete
from seer_annotator.llm import dummy_complete
from seer_annotator.rate_limiter import PerProviderRateLimiter
from seer_annotator.store import Store

console = Console()
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# BenchStore
# ---------------------------------------------------------------------------

class BenchStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._tx() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS format_bench (
                    run_id                 INTEGER NOT NULL,
                    paper_id               INTEGER NOT NULL,
                    version_id             INTEGER NOT NULL,
                    format_model           TEXT NOT NULL,
                    format_provider        TEXT,
                    value_json             TEXT,
                    cited_text_json        TEXT,
                    confidence             INTEGER,
                    citation_verified      INTEGER,
                    citation_verify_note   TEXT,
                    value_matches_ref      INTEGER,
                    cite_count_ref         INTEGER,
                    cite_count_test        INTEGER,
                    cite_similarity        REAL,
                    pass2_text             TEXT,
                    parse_error            TEXT,
                    fmt_tokens_input       INTEGER,
                    fmt_tokens_output      INTEGER,
                    fmt_tokens_cached      INTEGER,
                    fmt_cost               TEXT,
                    fmt_latency_ms         INTEGER,
                    created_at             TEXT NOT NULL,
                    PRIMARY KEY (run_id, paper_id, version_id, format_model)
                );
            """)

    def save_row(self, row: dict) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO format_bench (
                    run_id, paper_id, version_id, format_model, format_provider,
                    value_json, cited_text_json, confidence,
                    citation_verified, citation_verify_note,
                    value_matches_ref, cite_count_ref, cite_count_test, cite_similarity,
                    pass2_text, parse_error,
                    fmt_tokens_input, fmt_tokens_output, fmt_tokens_cached, fmt_cost, fmt_latency_ms,
                    created_at
                ) VALUES (
                    :run_id, :paper_id, :version_id, :format_model, :format_provider,
                    :value_json, :cited_text_json, :confidence,
                    :citation_verified, :citation_verify_note,
                    :value_matches_ref, :cite_count_ref, :cite_count_test, :cite_similarity,
                    :pass2_text, :parse_error,
                    :fmt_tokens_input, :fmt_tokens_output, :fmt_tokens_cached, :fmt_cost, :fmt_latency_ms,
                    :created_at
                )""",
                row,
            )

    def all_rows(
        self,
        run_ids: list[int] | None = None,
        paper_ids: list[int] | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM format_bench WHERE 1=1"
        params: list = []
        if run_ids is not None:
            placeholders = ",".join("?" * len(run_ids))
            query += f" AND run_id IN ({placeholders})"
            params.extend(run_ids)
        if paper_ids is not None:
            placeholders = ",".join("?" * len(paper_ids))
            query += f" AND paper_id IN ({placeholders})"
            params.extend(paper_ids)
        query += " ORDER BY format_model, run_id, paper_id, version_id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _count_citations(cited: object) -> int:
    """Number of citation segments: 0 for empty/null, 1 for string, len for list."""
    if not cited:
        return 0
    if isinstance(cited, list):
        return len([x for x in cited if x])
    return 1


def _normalise_text(cited: object) -> str:
    """Flatten cited_text to a single normalised string for similarity comparison."""
    if not cited:
        return ""
    if isinstance(cited, list):
        text = " ".join(str(x) for x in cited if x)
    else:
        text = str(cited)
    return " ".join(text.lower().split())


def _compare_cited(
    ref_cited: object,
    test_cited: object,
) -> tuple[int, int, float | None]:
    """Return (ref_count, test_count, similarity_ratio).

    similarity is None when the reference has no citation.
    """
    ref_count = _count_citations(ref_cited)
    test_count = _count_citations(test_cited)
    if ref_count == 0:
        return ref_count, test_count, None
    ref_norm = _normalise_text(ref_cited)
    test_norm = _normalise_text(test_cited)
    if not ref_norm:
        return ref_count, test_count, None
    similarity = difflib.SequenceMatcher(None, ref_norm, test_norm).ratio()
    return ref_count, test_count, similarity


def _compare_value(ref_value: object, test_value: object) -> bool:
    return json.dumps(ref_value, sort_keys=True) == json.dumps(test_value, sort_keys=True)


# ---------------------------------------------------------------------------
# Core bench logic
# ---------------------------------------------------------------------------

async def bench_pipeline(
    pipeline: PipelineConfig,
    settings: Settings,
    bench: BenchStore,
    models: list[tuple[str, str | None]],
    *,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
    dry_run: bool = False,
    concurrency: int | None = None,
) -> None:
    """Run each test model against every stored Pass-1 output and persist results."""
    import litellm as _litellm

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)
    _litellm.drop_params = pipeline_cfg.drop_params

    store = Store(settings.runtime.store_path)
    question_map = {q.version_id: q for q in pipeline.questions}
    question_order = {q.version_id: i for i, q in enumerate(pipeline.questions)}
    paper_map = {p.paper_id: p for p in pipeline.papers}

    runs = [r for r in pipeline.runs if run_ids is None or r.run_id in run_ids]
    papers = [p for p in pipeline.papers if paper_ids is None or p.paper_id in paper_ids]

    _concurrency = concurrency or pipeline_cfg.concurrency
    sem = asyncio.Semaphore(_concurrency)
    _base_complete = dummy_complete if dry_run else llm_complete
    _limiter = PerProviderRateLimiter(pipeline_cfg.per_provider_rpm)

    async def _rate_limited(model, provider, messages, **kw):
        await _limiter.acquire(provider)
        return await _base_complete(model, provider, messages, **kw)

    # Total work = runs × papers × models (for progress bar)
    total_work = len(runs) * len(papers) * len(models)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Benchmarking", total=total_work)

        for run in runs:
            cfg = effective_run_config(run.config, settings.run_defaults)

            for model_name, model_provider in models:
                final_provider = model_provider or run.model_provider
                bench_key = f"{final_provider}/{model_name}"

                if not dry_run:
                    prov_cfg = settings.providers.get(final_provider, ProviderSettings())
                    fmt_extra: dict = {}
                    api_key = prov_cfg.resolved_api_key()
                    if api_key:
                        fmt_extra["api_key"] = api_key
                    if prov_cfg.base_url:
                        fmt_extra["api_base"] = prov_cfg.base_url
                    if prov_cfg.api_version:
                        fmt_extra["api_version"] = prov_cfg.api_version
                    complete_fn = (
                        functools.partial(_rate_limited, **fmt_extra)
                        if fmt_extra
                        else _rate_limited
                    )
                else:
                    complete_fn = _rate_limited

                for paper in papers:
                    rows = store.get_reformattable_rows(run.run_id, paper.paper_id)
                    if not rows:
                        progress.advance(task)
                        continue

                    if cfg.text_source == "full_text":
                        source_text = store.get_ocr(paper.paper_id) or ""
                    else:
                        source_text = paper_map.get(paper.paper_id, paper).abstract

                    ref_map = {r["version_id"]: r["payload"] for r in rows}

                    groups: dict[str, list[dict]] = {}
                    for row in rows:
                        raw_resp = json.loads(row["payload"]["raw_response"])
                        group_id = raw_resp.get("batch_group_id") or f"solo_{row['version_id']}"
                        groups.setdefault(group_id, []).append(row)

                    progress.update(
                        task,
                        description=f"{bench_key} paper={paper.paper_id}",
                    )

                    async def _do_group(
                        group_id: str,
                        group_rows: list[dict],
                        *,
                        model_name: str = model_name,
                        final_provider: str = final_provider,
                        bench_key: str = bench_key,
                        complete_fn=complete_fn,
                        ref_map: dict = ref_map,
                        source_text: str = source_text,
                    ) -> None:
                        raw0 = json.loads(group_rows[0]["payload"]["raw_response"])
                        pass1_text = raw0.get("pass1_text")
                        if not pass1_text:
                            logger.warning(
                                "No pass1_text for run=%d paper=%d group=%s — skipping",
                                run.run_id, paper.paper_id, group_id,
                            )
                            for row in group_rows:
                                bench.save_row(_error_row(
                                    row, bench_key, final_provider,
                                    "no pass1_text in stored raw_response",
                                    ref_map,
                                ))
                            return

                        questions_in_group = sorted(
                            [
                                question_map[r["version_id"]]
                                for r in group_rows
                                if r["version_id"] in question_map
                            ],
                            key=lambda q: question_order.get(q.version_id, 0),
                        )
                        q_to_row = {r["version_id"]: r for r in group_rows}
                        ordered_rows = [q_to_row[q.version_id] for q in questions_in_group]

                        async with sem:
                            t0 = time.monotonic()
                            try:
                                results = await reformat_group(
                                    pass1_text=pass1_text,
                                    source_text=source_text,
                                    questions=questions_in_group,
                                    format_model=model_name,
                                    format_model_provider=final_provider,
                                    format_structured_output=cfg.format_structured_output,
                                    format_model_params=cfg.format_model_params,
                                    complete_fn=complete_fn,
                                    citation_max_error_rate=cfg.citation_max_error_rate,
                                    citation_max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                                )
                                group_latency_ms = int((time.monotonic() - t0) * 1000)
                            except ExtractionError as exc:
                                logger.error(
                                    "ExtractionError %s run=%d paper=%d group=%s: %s",
                                    bench_key, run.run_id, paper.paper_id, group_id, exc,
                                )
                                for row in ordered_rows:
                                    bench.save_row(_error_row(
                                        row, bench_key, final_provider,
                                        f"ExtractionError: {exc}",
                                        ref_map,
                                    ))
                                return
                            except Exception as exc:
                                logger.error(
                                    "Failed %s run=%d paper=%d group=%s: %s",
                                    bench_key, run.run_id, paper.paper_id, group_id, exc,
                                )
                                for row in ordered_rows:
                                    bench.save_row(_error_row(
                                        row, bench_key, final_provider,
                                        f"{type(exc).__name__}: {exc}",
                                        ref_map,
                                    ))
                                return

                        for i, (row, result) in enumerate(zip(ordered_rows, results)):
                            ref_payload = ref_map[row["version_id"]]
                            ref_raw = json.loads(ref_payload["raw_response"])
                            ref_value = ref_raw.get("parse_result", {}).get("value")
                            ref_cited = ref_payload.get("cited_text")

                            parsed = result["parse_result"]
                            test_value = parsed.get("value")
                            test_cited = parsed.get("cited_text")

                            ref_count, test_count, similarity = _compare_cited(ref_cited, test_cited)

                            cited_verified = result["cited_text_verified"]

                            bench.save_row({
                                "run_id": row["run_id"],
                                "paper_id": row["paper_id"],
                                "version_id": row["version_id"],
                                "format_model": bench_key,
                                "format_provider": final_provider,
                                "value_json": json.dumps(test_value),
                                "cited_text_json": json.dumps(test_cited),
                                "confidence": parsed.get("confidence"),
                                "citation_verified": (
                                    int(cited_verified) if cited_verified is not None else None
                                ),
                                "citation_verify_note": result["verify"].get("note"),
                                "value_matches_ref": int(_compare_value(ref_value, test_value)),
                                "cite_count_ref": ref_count,
                                "cite_count_test": test_count,
                                "cite_similarity": similarity,
                                "pass2_text": result["p2_text"],
                                "parse_error": None,
                                "fmt_tokens_input": result["fmt_tokens_input"],
                                "fmt_tokens_output": result["fmt_tokens_output"],
                                "fmt_tokens_cached": result["fmt_tokens_cached"],
                                "fmt_cost": (
                                    str(result["fmt_cost"]) if result["fmt_cost"] else None
                                ),
                                # Latency attributed to first answer in group (mirrors annotate_group)
                                "fmt_latency_ms": group_latency_ms if i == 0 else 0,
                                "created_at": _now(),
                            })

                    await asyncio.gather(
                        *[_do_group(gid, grows) for gid, grows in groups.items()]
                    )
                    progress.advance(task)


def _error_row(
    row: dict,
    bench_key: str,
    final_provider: str,
    error_msg: str,
    ref_map: dict,
) -> dict:
    ref_payload = ref_map.get(row["version_id"], {})
    ref_raw = json.loads(ref_payload.get("raw_response") or "{}")
    ref_value = ref_raw.get("parse_result", {}).get("value")
    ref_cited = ref_payload.get("cited_text")
    ref_count = _count_citations(ref_cited)
    return {
        "run_id": row["run_id"],
        "paper_id": row["paper_id"],
        "version_id": row["version_id"],
        "format_model": bench_key,
        "format_provider": final_provider,
        "value_json": None,
        "cited_text_json": None,
        "confidence": None,
        "citation_verified": None,
        "citation_verify_note": None,
        "value_matches_ref": None,
        "cite_count_ref": ref_count,
        "cite_count_test": None,
        "cite_similarity": None,
        "pass2_text": None,
        "parse_error": error_msg,
        "fmt_tokens_input": None,
        "fmt_tokens_output": None,
        "fmt_tokens_cached": None,
        "fmt_cost": None,
        "fmt_latency_ms": None,
        "created_at": _now(),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.1f}%"


def _fmt_cost(total: float) -> str:
    if total == 0.0:
        return "$0.000000"
    if total < 0.001:
        return f"${total:.6f}"
    return f"${total:.4f}"


def print_report(
    bench: BenchStore,
    pipeline: PipelineConfig,
    store: Store,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
    detail: bool = False,
) -> None:
    rows = bench.all_rows(run_ids=run_ids, paper_ids=paper_ids)
    if not rows:
        console.print("[yellow]No bench results found.[/yellow]")
        return

    # Build reference stats from main answers table
    all_ref_rows = store.all_answers()
    if run_ids:
        all_ref_rows = [r for r in all_ref_rows if r["run_id"] in run_ids]
    if paper_ids:
        all_ref_rows = [r for r in all_ref_rows if r["paper_id"] in paper_ids]
    ref_with_payload = [r for r in all_ref_rows if r.get("payload_json")]

    ref_total = len(ref_with_payload)

    # Citation verification rate for the reference model
    ref_cite_verified_n = 0
    ref_cite_with_citation_n = 0
    for r in ref_with_payload:
        payload = json.loads(r["payload_json"])
        cited = payload.get("cited_text")
        verified = payload.get("cited_text_verified")
        if cited:
            ref_cite_with_citation_n += 1
            if verified:
                ref_cite_verified_n += 1

    ref_cost_total = sum(
        float(json.loads(r["payload_json"]).get("fmt_cost") or 0)
        for r in ref_with_payload
    )

    # Determine reference model label from stored p2_raw if possible
    ref_model_label = "[reference]"
    if ref_with_payload:
        try:
            raw = json.loads(json.loads(ref_with_payload[0]["payload_json"])["raw_response"])
            p2_raw = raw.get("p2_raw") or {}
            if isinstance(p2_raw, dict) and p2_raw.get("model"):
                ref_model_label = f"[ref] {p2_raw['model']}"
        except Exception:
            pass

    # Aggregate bench rows by format_model
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["format_model"], []).append(row)

    # Summary table
    n_runs = len({r["run_id"] for r in rows})
    n_papers = len({r["paper_id"] for r in rows})
    n_questions = len({r["version_id"] for r in rows})
    console.print(
        f"\n[bold]Format Model Benchmark[/bold] — "
        f"{n_runs} run(s), {n_papers} paper(s), {n_questions} question version(s), "
        f"{ref_total} reference answers\n"
    )

    table = Table(show_header=True, header_style="bold cyan", show_lines=False)
    table.add_column("Model", style="bold", min_width=30)
    table.add_column("Answers", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Value Match", justify="right")
    table.add_column("Cite Count ✓", justify="right")
    table.add_column("Cite Verified", justify="right")
    table.add_column("Cite Similarity", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Cost/Answer", justify="right")
    table.add_column("Total Latency", justify="right")
    table.add_column("Avg Latency/Call", justify="right")

    ref_cite_verified_pct = _pct(ref_cite_verified_n, ref_cite_with_citation_n)

    # Reference latency from main answers table
    ref_latency_ms_vals = []
    for r in ref_with_payload:
        payload = json.loads(r["payload_json"])
        lat = payload.get("fmt_latency_ms")
        if lat:
            ref_latency_ms_vals.append(lat)
    ref_total_latency = sum(ref_latency_ms_vals)
    ref_avg_latency = (ref_total_latency / len(ref_latency_ms_vals)) if ref_latency_ms_vals else None

    def _fmt_latency(ms: float | None) -> str:
        if ms is None:
            return "n/a"
        if ms >= 60_000:
            return f"{ms / 60_000:.1f}m"
        if ms >= 1_000:
            return f"{ms / 1_000:.1f}s"
        return f"{ms:.0f}ms"

    table.add_row(
        ref_model_label,
        str(ref_total),
        "—",
        "(baseline)",
        "(baseline)",
        f"{ref_cite_verified_pct} (baseline)",
        "(baseline)",
        _fmt_cost(ref_cost_total),
        _fmt_cost(ref_cost_total / ref_total) if ref_total else "n/a",
        _fmt_latency(ref_total_latency) if ref_latency_ms_vals else "n/a",
        _fmt_latency(ref_avg_latency),
        style="dim",
    )

    for model_key in sorted(by_model.keys()):
        model_rows = by_model[model_key]
        n_total = len(model_rows)
        n_errors = sum(1 for r in model_rows if r.get("parse_error"))
        n_ok = n_total - n_errors

        value_matches = sum(
            1 for r in model_rows if r.get("parse_error") is None and r.get("value_matches_ref") == 1
        )
        cite_count_match = sum(
            1 for r in model_rows
            if r.get("parse_error") is None
            and r.get("cite_count_ref") is not None
            and r.get("cite_count_test") is not None
            and r["cite_count_ref"] == r["cite_count_test"]
        )
        # Only count answers where reference had a citation (cite_count_ref > 0)
        rows_with_ref_cite = [
            r for r in model_rows
            if r.get("parse_error") is None and (r.get("cite_count_ref") or 0) > 0
        ]
        cite_verified_ok = sum(
            1 for r in rows_with_ref_cite if r.get("citation_verified") == 1
        )
        similarity_vals = [
            r["cite_similarity"]
            for r in rows_with_ref_cite
            if r.get("cite_similarity") is not None
        ]
        mean_similarity = (
            f"{100 * sum(similarity_vals) / len(similarity_vals):.1f}%"
            if similarity_vals
            else "n/a"
        )

        cost_total = sum(float(r["fmt_cost"]) for r in model_rows if r.get("fmt_cost"))

        # Latency: only non-zero values represent real call latencies (0 = non-first in group)
        latency_vals = [r["fmt_latency_ms"] for r in model_rows if r.get("fmt_latency_ms")]
        total_latency_ms = sum(latency_vals)
        avg_latency_ms = total_latency_ms / len(latency_vals) if latency_vals else None

        # Flag models that look problematic
        cite_count_ok_pct = (cite_count_match / n_ok * 100) if n_ok else 0
        cite_verified_ok_pct = (
            (cite_verified_ok / len(rows_with_ref_cite) * 100) if rows_with_ref_cite else 0
        )
        row_style = ""
        if n_ok > 0 and (cite_count_ok_pct < 99.0 or (
            ref_cite_with_citation_n > 0
            and abs(cite_verified_ok_pct - (100 * ref_cite_verified_n / ref_cite_with_citation_n)) > 2.0
        )):
            row_style = "yellow"

        table.add_row(
            model_key,
            str(n_total),
            str(n_errors) if n_errors else "—",
            _pct(value_matches, n_ok),
            _pct(cite_count_match, n_ok),
            _pct(cite_verified_ok, len(rows_with_ref_cite)) if rows_with_ref_cite else "n/a",
            mean_similarity,
            _fmt_cost(cost_total),
            _fmt_cost(cost_total / n_total) if n_total else "n/a",
            _fmt_latency(total_latency_ms) if latency_vals else "n/a",
            _fmt_latency(avg_latency_ms),
            style=row_style,
        )

    console.print(table)
    console.print(
        "[dim]Yellow rows: cite count match < 99% or cite verification rate "
        "deviates > 2 pp from reference[/dim]\n"
    )

    if not detail:
        return

    # Per-question-key breakdown
    question_map = {q.version_id: q for q in pipeline.questions}
    console.print("\n[bold]Per-question breakdown[/bold]\n")

    all_version_ids = sorted({r["version_id"] for r in rows})
    for vid in all_version_ids:
        q = question_map.get(vid)
        q_label = f"{q.key} (v{q.version})" if q else f"version_id={vid}"
        console.print(f"  [cyan]{q_label}[/cyan]")

        q_table = Table(show_header=True, header_style="bold", show_lines=False, padding=(0, 1))
        q_table.add_column("Model")
        q_table.add_column("Value Match", justify="right")
        q_table.add_column("Cite Count ✓", justify="right")
        q_table.add_column("Cite Verified", justify="right")
        q_table.add_column("Cite Similarity", justify="right")
        q_table.add_column("Errors", justify="right")

        for model_key in sorted(by_model.keys()):
            q_rows = [r for r in by_model[model_key] if r["version_id"] == vid]
            if not q_rows:
                continue
            n_err = sum(1 for r in q_rows if r.get("parse_error"))
            n_ok = len(q_rows) - n_err
            vm = sum(1 for r in q_rows if not r.get("parse_error") and r.get("value_matches_ref") == 1)
            ccm = sum(
                1 for r in q_rows
                if not r.get("parse_error")
                and r.get("cite_count_ref") is not None
                and r.get("cite_count_test") is not None
                and r["cite_count_ref"] == r["cite_count_test"]
            )
            q_with_cite = [r for r in q_rows if not r.get("parse_error") and (r.get("cite_count_ref") or 0) > 0]
            cv = sum(1 for r in q_with_cite if r.get("citation_verified") == 1)
            sims = [r["cite_similarity"] for r in q_with_cite if r.get("cite_similarity") is not None]
            mean_sim = f"{100 * sum(sims) / len(sims):.1f}%" if sims else "n/a"

            q_table.add_row(
                model_key,
                _pct(vm, n_ok),
                _pct(ccm, n_ok),
                _pct(cv, len(q_with_cite)) if q_with_cite else "n/a",
                mean_sim,
                str(n_err) if n_err else "—",
            )

        console.print(q_table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_model(value: str) -> tuple[str, str | None]:
    """Split 'provider/model' into (model, provider). Provider is None if absent."""
    if "/" in value:
        provider, _, model = value.partition("/")
        return model, provider
    return value, None


@click.command()
@click.argument("pipeline_json", type=click.Path(exists=True))
@click.option(
    "--model", "model_specs", multiple=True, required=True,
    metavar="PROVIDER/MODEL",
    help="Format model to benchmark, in litellm provider/model format (repeatable). "
         "E.g. --model openai/gpt-4o-mini --model anthropic/claude-haiku-4-5-20251001. "
         "Omit provider to inherit from the run's model_provider.",
)
@click.option("--runs", default=None, help="Comma-separated run IDs to include.")
@click.option("--papers", default=None, help="Comma-separated paper IDs to include.")
@click.option("--settings", "settings_path", default=None,
              help="Path to settings.toml (default: settings.toml / $SEER_SETTINGS).")
@click.option("--bench-db", "bench_db", default=None,
              help="SQLite file for bench results. Defaults to the main answers DB.")
@click.option("--concurrency", default=None, type=int,
              help="Override parallel Pass-2 calls (default: from settings).")
@click.option("--no-structured-output", "no_structured_output", is_flag=True,
              help="Disable JSON schema response_format (for models that don't support it).")
@click.option("--dry-run", is_flag=True,
              help="Use dummy LLM, no real API calls. Useful for verifying DB schema.")
@click.option("--detail", is_flag=True,
              help="Print per-question breakdown in addition to the summary.")
@click.option("--report-only", is_flag=True,
              help="Skip benchmarking, just print the report from existing bench results.")
@click.option("--verbose", "-v", is_flag=True)
def cli(
    pipeline_json: str,
    model_specs: tuple[str, ...],
    runs: str | None,
    papers: str | None,
    settings_path: str | None,
    bench_db: str | None,
    concurrency: int | None,
    no_structured_output: bool,
    dry_run: bool,
    detail: bool,
    report_only: bool,
    verbose: bool,
) -> None:
    """Benchmark multiple Pass-2 formatting models against stored Pass-1 outputs.

    Results go into a ``format_bench`` table (separate from ``answers``).
    The main DB is never modified.

    \b
    Example:
        python bench_format_models.py pipeline.json \\
            --model openai/gpt-4o-mini \\
            --model anthropic/claude-haiku-4-5-20251001
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    with open(pipeline_json) as f:
        pipeline = PipelineConfig.model_validate(json.load(f))

    settings = Settings.load(settings_path)

    run_ids = [int(x) for x in runs.split(",")] if runs else None
    paper_ids = [int(x) for x in papers.split(",")] if papers else None

    db_path = bench_db or settings.runtime.store_path
    bench = BenchStore(db_path)
    store = Store(settings.runtime.store_path)

    models = [_parse_model(s) for s in model_specs]

    if not report_only:
        # Patch format_structured_output into settings.run_defaults if --no-structured-output
        if no_structured_output:
            settings.run_defaults.format_structured_output = False

        asyncio.run(
            bench_pipeline(
                pipeline,
                settings,
                bench,
                models,
                run_ids=run_ids,
                paper_ids=paper_ids,
                dry_run=dry_run,
                concurrency=concurrency,
            )
        )

    print_report(
        bench,
        pipeline,
        store,
        run_ids=run_ids,
        paper_ids=paper_ids,
        detail=detail,
    )


if __name__ == "__main__":
    cli()
