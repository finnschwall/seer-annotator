"""Read-only source database adapter and snapshotter.

This module is the only benchmark code allowed to know about a SEER checkout.
It uses Django's ORM for discovery, but never imports SEER pipeline/job helpers
(those helpers can create jobs, tokens, or files).  All archive files are
opened read-only and only an allowlisted subset is copied into SQLite.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import os
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..batching import resolve_groups
from ..config import Question, RunConfig
from .store import BenchmarkCase, BenchmarkStore, DatasetSpec

logger = logging.getLogger(__name__)


class SourceSafetyError(RuntimeError):
    """The source connection or archive failed a read-only safety check."""


def configure_postgres_read_only_options(connection: Any) -> None:
    """Inject the PostgreSQL read-only startup option before opening a socket.

    Django's PostgreSQL backend passes ``OPTIONS['options']`` to libpq.  This
    connection-level setting closes the race between connection creation and
    the first SQL ``SET`` statement; the latter remains below as an assertion
    and transaction-level defense.
    """
    if getattr(connection, "vendor", "postgresql") != "postgresql":
        raise SourceSafetyError("source database must use PostgreSQL; refusing a SQLite fallback")
    settings = getattr(connection, "settings_dict", None)
    if settings is None:
        raise SourceSafetyError("source connection has no configurable database settings")
    options = dict(settings.get("OPTIONS") or {})
    existing = str(options.get("options") or "").strip()
    marker = "default_transaction_read_only=on"
    if marker not in existing.replace(" ", ""):
        options["options"] = (existing + " " if existing else "") + "-c default_transaction_read_only=on"
    settings["OPTIONS"] = options
    setattr(connection, "_benchmark_read_only_configured", True)


@dataclass(frozen=True)
class SourceRun:
    run_id: int
    name: str = ""
    model_name: str = ""
    model_provider: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    pipeline_path: str | None = None
    usable_papers: int = 0
    reason: str = ""


@contextmanager
def source_logging_disabled() -> Iterator[None]:
    """Prevent source checkout handlers from writing during an import."""
    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(old_disable)


@contextmanager
def postgres_read_only(connection: Any) -> Iterator[Any]:
    """Yield a Django connection in an asserted read-only transaction.

    ``ensure_connection`` is intentionally called before any ORM operation.
    The session default is forced first, then the transaction is explicitly
    marked read-only and both settings are checked.  A non-PostgreSQL backend
    is rejected: a benchmark source must not accidentally use a local fallback
    SQLite database.
    """
    configure_postgres_read_only_options(connection)
    connection.ensure_connection()
    with connection.cursor() as cursor:
        cursor.execute("SET default_transaction_read_only = on")
        cursor.execute("SHOW default_transaction_read_only")
        default_value = str(cursor.fetchone()[0]).lower()
    if default_value != "on":
        raise SourceSafetyError("could not verify PostgreSQL default_transaction_read_only=on")
    # Django's atomic block issues BEGIN before its body, making SET TRANSACTION
    # legal and ensuring every ORM query in the block is protected.
    from django.db import transaction
    with transaction.atomic(using=getattr(connection, "alias", "default")):
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SHOW transaction_read_only")
            tx_value = str(cursor.fetchone()[0]).lower()
        if tx_value != "on":
            raise SourceSafetyError("could not verify PostgreSQL transaction_read_only=on")
        setattr(connection, "_benchmark_read_only_verified", True)
        yield connection


def _env_from_settings_ini(root: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    path = root / "settings.ini"
    if not path.exists():
        raise SourceSafetyError(f"source settings file does not exist: {path}")
    parser.read(path)
    section = parser["settings"] if parser.has_section("settings") else parser.defaults()
    # Never return or log unrelated settings (which may contain API keys).
    names = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT")
    return {name: section.get(name, "").strip().strip('"') for name in names if section.get(name)}


@contextmanager
def source_environment(root: Path) -> Iterator[None]:
    """Install source settings only while Django initializes, then restore env."""
    updates = {**_env_from_settings_ini(root), "DJANGO_SETTINGS_MODULE": "systematic_review.settings", "LOG_FILE": ""}
    sentinel = object()
    previous = {key: os.environ.get(key, sentinel) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value  # type: ignore[assignment]


class DjangoSource:
    """Minimal, read-only ORM adapter for a configured SEER checkout."""

    def __init__(self, seer_root: str | Path):
        self.root = Path(seer_root).resolve()
        if not self.root.is_dir():
            raise SourceSafetyError(f"source checkout does not exist: {self.root}")
        self.read_only_configured = False
        self.read_only_verified = False
        self._setup_django()

    def _setup_django(self) -> None:
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        # settings.py constructs the RotatingFileHandler while Django starts;
        # source_environment sets LOG_FILE='' before setup and restores ambient
        # settings afterward.
        with source_environment(self.root):
            with source_logging_disabled():
                import django
                django.setup()
        from django.db import connections
        self.connection = connections["default"]
        configure_postgres_read_only_options(self.connection)
        self.read_only_configured = True

    @contextmanager
    def read_only(self) -> Iterator[None]:
        with source_logging_disabled(), postgres_read_only(self.connection):
            self.read_only_verified = True
            yield

    def list_runs(self) -> list[SourceRun]:
        from experiments.models import ExperimentRun, LLMCallTrace, LLMJob
        with self.read_only():
            runs = list(ExperimentRun.objects.all().order_by("pk"))
            result = []
            for run in runs:
                traces = list(LLMCallTrace.objects.filter(run_id=run.pk, pass1_outcome="ok")
                              .exclude(pass1_text="").select_related("job"))
                cache: dict[str, dict[str, Any]] = {}
                source_run = SourceRun(run.pk, config=dict(run.config or {}))
                usable_ids = set()
                for trace in traces:
                    trace_data = {"paper_id": trace.paper_id, "group_id": trace.group_id,
                                  "question_keys": list(trace.question_keys or []),
                                  "pipeline_path": (str(next((p for p in self._job_pipeline_candidates(trace.job) if p.is_file()), ""))
                                                     if trace.job else None)}
                    loaded = _load_pipeline_for_trace(trace_data, cache)
                    if loaded and _trace_group(loaded[0], trace_data, source_run):
                        usable_ids.add(trace.paper_id)
                usable = len(usable_ids)
                jobs = LLMJob.objects.filter(run_id=run.pk).order_by("-created_at")
                pipeline_path = next((str(candidate) for j in jobs
                                      for candidate in self._job_pipeline_candidates(j)
                                      if candidate.is_file()), None)
                reason = "" if usable and pipeline_path else ("no successful Pass-1 traces" if not usable else "no archived pipeline.json")
                result.append(SourceRun(run.pk, run.name, run.model_name, run.model_provider,
                                        dict(run.config or {}), pipeline_path, usable, reason))
            return result

    def _archive_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _job_pipeline_candidates(self, job: Any) -> list[Path]:
        candidates: list[Path] = []
        if getattr(job, "pipeline_path", None):
            candidates.append(self._archive_path(job.pipeline_path))
        if getattr(job, "artifacts_dir", None):
            candidates.append(self._archive_path(job.artifacts_dir) / "pipeline.json")
        return candidates

    def traces_for_run(self, run_id: int) -> list[dict[str, Any]]:
        from experiments.models import LLMCallTrace
        with self.read_only():
            traces = list(
                LLMCallTrace.objects.filter(run_id=run_id, pass1_outcome="ok")
                .exclude(pass1_text="").select_related("paper", "job").order_by("paper_id", "-updated_at", "-pk")
            )
            return [{"id": t.pk, "paper_id": t.paper_id, "group_id": t.group_id,
                     "question_keys": list(t.question_keys or []), "pass1_text": t.pass1_text,
                     "paper_title": t.paper.title, "abstract": t.paper.abstract or "",
                     "pipeline_path": (str(next((p for p in self._job_pipeline_candidates(t.job) if p.is_file()), ""))
                                       if t.job else None),
                     "state_path": (str(self._archive_path(t.job.store_path)) if t.job and t.job.store_path else
                                    (str(self._archive_path(t.job.artifacts_dir) / "state.db")
                                     if t.job and t.job.artifacts_dir else None)),
                     "updated_at": t.updated_at.isoformat() if t.updated_at else ""} for t in traces]

    def pipeline_for_run(self, run_id: int) -> tuple[dict[str, Any], str]:
        from experiments.models import LLMJob
        with self.read_only():
            jobs = list(LLMJob.objects.filter(run_id=run_id).order_by("-created_at"))
        for job in jobs:
            for path in self._job_pipeline_candidates(job):
                if path.is_file():
                    try:
                        return json.loads(path.read_text(encoding="utf-8")), str(path)
                    except (OSError, json.JSONDecodeError) as exc:
                        raise SourceSafetyError(f"invalid archived pipeline {path}: {exc}") from exc
        raise SourceSafetyError(f"run {run_id} has no readable archived pipeline.json")

    def full_text_from_state(self, state_path: str | Path, paper_id: int) -> str | None:
        path = Path(state_path)
        if not path.is_file():
            return None
        # URI mode=ro prevents sqlite from creating or journal-writing a source file.
        uri = f"file:{path.resolve()}?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True)
            try:
                rows = con.execute(
                    "SELECT markdown FROM ocr_cache WHERE paper_id=?", (paper_id,)
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise SourceSafetyError(f"could not read archived state database read-only: {path}: {exc}") from exc
        if len(rows) > 1:
            # One store can hold several renderings of a paper — a job driving
            # runs that withhold different sections caches one per rendering.
            # Nothing here says which of them this run was sent, and quietly
            # picking one would put text into a benchmark that the run never saw.
            raise SourceSafetyError(
                f"archived state {path} holds {len(rows)} renderings of paper {paper_id}; "
                "cannot say which this run was sent"
            )
        return rows[0][0] if rows else None

    def current_source_text(self, paper_id: int) -> str | None:
        """Read the currently registered OCR markdown, only for explicit override."""
        from papers.models import PaperChunks
        with self.read_only():
            row = PaperChunks.objects.filter(paper_id=paper_id).values_list("markdown_path", flat=True).first()
        if not row:
            return None
        path = Path(row)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SourceSafetyError(f"could not read current OCR source {path}: {exc}") from exc


def stable_sample(paper_ids: Sequence[int], run_id: int, count: int, seed: int) -> list[int]:
    """Select a deterministic sample independently for one source run."""
    if count < 0:
        raise ValueError("paper count must be non-negative")
    scored = sorted((hashlib.sha256(f"{seed}:{run_id}:{pid}".encode()).hexdigest(), int(pid)) for pid in set(paper_ids))
    return [pid for _, pid in scored[:count]]


def _questions_from_pipeline(raw: dict[str, Any]) -> list[Question]:
    return [Question.model_validate(q) for q in raw.get("questions", [])]


def _expected_groups(raw: dict[str, Any], source_run: SourceRun) -> list[list[Question]]:
    run_data = next((r for r in raw.get("runs", []) if int(r.get("run_id")) == source_run.run_id), None)
    config = dict(source_run.config)
    if run_data:
        config = {**config, **dict(run_data.get("config") or {})}
    return resolve_groups(RunConfig.model_validate(config), _questions_from_pipeline(raw))


def _trace_group_index(group_id: str, run_id: int, paper_id: int) -> int | None:
    prefix = f"{run_id}-{paper_id}-"
    if group_id.startswith(prefix):
        try:
            return int(group_id[len(prefix):])
        except ValueError:
            return None
    return None


def _load_pipeline_for_trace(trace: dict[str, Any], cache: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    """Load exactly the archive attached to a trace/job; never fall back to a run's latest job."""
    embedded = trace.get("pipeline")
    if isinstance(embedded, dict):
        return embedded, str(trace.get("pipeline_path") or "<embedded>")
    raw_path = trace.get("pipeline_path")
    if not raw_path:
        return None
    path = str(raw_path)
    if path in cache:
        return cache[path], path
    archive = Path(path)
    if not archive.is_file():
        return None
    try:
        raw = json.loads(archive.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    cache[path] = raw
    return raw, path


def _trace_group(raw: dict[str, Any], trace: dict[str, Any], source_run: SourceRun) -> tuple[dict[str, Any], list[Question]] | None:
    try:
        groups = _expected_groups(raw, SourceRun(source_run.run_id, config=dict(trace.get("run_config") or source_run.config)))
    except (TypeError, ValueError, KeyError):
        return None
    paper_id = int(trace["paper_id"])
    paper = next((p for p in raw.get("papers", []) if int(p.get("paper_id")) == paper_id), None)
    if not paper:
        return None
    idx = _trace_group_index(str(trace.get("group_id", "")), source_run.run_id, paper_id)
    question_group = groups[idx] if idx is not None and idx < len(groups) else None
    if question_group is None:
        keys = set(trace.get("question_keys") or [])
        question_group = next((g for g in groups if keys and keys.intersection(q.key for q in g)), None)
    if not question_group:
        return None
    return paper, question_group


class SourceSnapshotter:
    """Copy selected source runs into a local :class:`BenchmarkStore`.

    ``adapter`` is injectable for tests and for alternate source databases. It
    must implement ``list_runs``, ``traces_for_run`` and ``pipeline_for_run``;
    ``DjangoSource`` supplies the production implementation.
    """

    def __init__(self, adapter: Any, store: BenchmarkStore):
        self.adapter, self.store = adapter, store

    def import_dataset(self, spec: DatasetSpec, runs: Sequence[tuple[int, int | None]], *, allow_current_source: bool = False) -> dict[str, int]:
        self.store.create_dataset(spec)
        available = {r.run_id: r for r in self.adapter.list_runs()}
        imported = 0
        for run_id, limit in runs:
            if run_id not in available:
                raise SourceSafetyError(f"source run {run_id} does not exist")
            source_run = available[run_id]
            traces = self.adapter.traces_for_run(run_id)
            pipeline_cache: dict[str, dict[str, Any]] = {}
            valid: list[tuple[dict[str, Any], dict[str, Any], list[Question], str]] = []
            for trace in traces:
                loaded = _load_pipeline_for_trace(trace, pipeline_cache)
                if not loaded:
                    continue
                grouped = _trace_group(loaded[0], trace, source_run)
                if grouped:
                    valid.append((trace, grouped[0], grouped[1], loaded[1]))
            # For full-text datasets, source availability is part of paper
            # eligibility.  Filtering after deterministic sampling could select
            # five papers but import only three, making ``RUN:5`` unreliable.
            # Probe each paper once before sampling and retain the frozen text
            # for the later case construction.
            full_text_by_paper: dict[int, tuple[str, str]] = {}
            if spec.case_kind == "full_text":
                for trace, paper, _question_group, _pipeline_path in valid:
                    paper_id = int(trace["paper_id"])
                    if paper_id in full_text_by_paper:
                        continue
                    source_text = None
                    state_path = trace.get("state_path")
                    if state_path:
                        source_text = self.adapter.full_text_from_state(state_path, paper_id)
                    text_kind = "full_text"
                    if source_text is None and allow_current_source:
                        current_reader = getattr(self.adapter, "current_source_text", None)
                        source_text = current_reader(paper_id) if current_reader else None
                        if source_text is not None:
                            text_kind = "current_full_text"
                    if source_text is not None:
                        full_text_by_paper[paper_id] = (str(source_text), text_kind)
                valid = [item for item in valid if int(item[0]["paper_id"]) in full_text_by_paper]
            valid_papers = sorted({int(trace["paper_id"]) for trace, _, _, _ in valid})
            requested = len(valid_papers) if limit is None else limit
            if requested < 0:
                raise ValueError("paper count must be non-negative")
            if len(valid_papers) < requested:
                if spec.case_kind == "full_text":
                    raise SourceSafetyError(f"run {run_id} produced only {len(valid_papers)} cases for {requested} requested papers")
                raise SourceSafetyError(f"run {run_id} has only {len(valid_papers)} importable papers; requested {requested}")
            selected_ids = set(stable_sample(valid_papers, run_id, requested, spec.seed))
            self.store.add_source_run(spec.name, source_run_id=run_id, name=source_run.name,
                                      model_name=source_run.model_name, model_provider=source_run.model_provider,
                                      config=source_run.config, pipeline_path=(source_run.pipeline_path or ""),
                                      usable_papers=len(valid_papers),
                                      provenance={"eligibility_reason": source_run.reason})
            imported_papers: set[int] = set()
            # ORM ordering is newest-first. Keep one successful trace per
            # logical group, so retries never silently create duplicate cases.
            seen_groups: set[tuple[int, str]] = set()
            for trace, paper, question_group, pipeline_path in valid:
                paper_id = int(trace["paper_id"])
                if paper_id not in selected_ids:
                    continue
                group_key = str(trace.get("group_id") or "")
                if not group_key:
                    group_key = "keys:" + ",".join(trace.get("question_keys") or [])
                if (paper_id, group_key) in seen_groups:
                    continue
                seen_groups.add((paper_id, group_key))
                # ORM ordering is newest-first. Keep one successful trace per
                # logical group, so retries never silently create duplicate
                # benchmark cases or make the selected Pass-1 text unstable.
                text_kind = spec.case_kind
                source_text = str(paper.get("abstract") or "")
                if text_kind == "full_text":
                    source_text, text_kind = full_text_by_paper[paper_id]
                case_key = f"{run_id}:{paper_id}:{trace.get('group_id') or trace.get('id')}"
                case = BenchmarkCase(
                        dataset=spec.name, case_key=case_key, source_run_id=run_id,
                        source_paper_id=paper_id, source_trace_id=trace.get("id"),
                        source_group_id=str(trace.get("group_id") or ""),
                        paper_title=str(paper.get("title") or trace.get("paper_title") or ""),
                        source_text=source_text, text_kind=text_kind,
                        pass1_text=str(trace.get("pass1_text") or ""),
                        questions=[q.model_dump(mode="json") for q in question_group],
                        provenance={"source_root": spec.source_root, "pipeline_path": pipeline_path,
                                    "source_trace_id": trace.get("id"), "source_checksum": hashlib.sha256(source_text.encode()).hexdigest()},
                )
                self.store.add_case(case)
                imported += 1
                imported_papers.add(paper_id)
            if len(imported_papers) < requested:
                raise SourceSafetyError(f"run {run_id} produced only {len(imported_papers)} cases for {requested} requested papers")
        return {"runs": len(runs), "cases": imported}
