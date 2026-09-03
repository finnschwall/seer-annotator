"""Versioned SQLite store for reproducible, offline formatting benchmarks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_SECRET_EXACT = {
    "token", "access_token", "auth_token", "api_token", "api_key", "apikey",
    "password", "secret", "credential", "credentials",
}
_SECRET_SUFFIXES = (
    "_token", "_api_key", "_apikey", "_password", "_secret", "_credential", "_credentials",
)


def is_secret_key(key: object) -> bool:
    """Return whether a mapping key is a credential field.

    Deliberately do not use substring matching: ``max_tokens``,
    ``input_tokens`` and similar usage/configuration metrics are safe data.
    """
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SECRET_EXACT or normalized.endswith(_SECRET_SUFFIXES)


def redact_secrets(value: Any) -> Any:
    """Return JSON-compatible data with credential-looking fields removed."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if is_secret_key(key):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = redact_secrets(item)
        return result
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_root: str
    selection: Mapping[str, Any]
    source_kind: str = "seer"
    case_kind: str = "abstract"
    seed: int = 0

    @property
    def selection_json(self) -> str:
        return canonical_json(redact_secrets(dict(self.selection)))

    @property
    def spec_hash(self) -> str:
        return fingerprint({"source_root": self.source_root, "source_kind": self.source_kind,
                            "case_kind": self.case_kind, "seed": self.seed,
                            "selection": dict(self.selection)})


@dataclass(frozen=True)
class BenchmarkCase:
    dataset: str
    case_key: str
    source_run_id: int
    source_paper_id: int
    source_trace_id: int | None
    source_group_id: str
    paper_title: str
    source_text: str
    text_kind: str
    pass1_text: str
    questions: list[dict[str, Any]]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    original_answers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    structured_output: bool = True
    temperature: float | None = 0.0
    timeout: float | None = None
    drop_params: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return fingerprint(self.as_dict())


class BenchmarkStore:
    """SQLite persistence boundary for all benchmark stages.

    Dataset specifications are immutable: reusing a name with another
    specification raises ``ValueError``. Cases and source runs are upserted so
    a safely interrupted snapshot can be resumed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        # WAL lets independent runner processes write concurrently (each model
        # run claims its own execution rows), instead of serializing on a single
        # writer lock. The setting is persistent in the database file.
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _migrate(self) -> None:
        with self.transaction() as con:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"benchmark database schema {version} is newer than supported {SCHEMA_VERSION}")
            if version < 1:
                con.executescript(
                    """
                    CREATE TABLE datasets (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        source_root TEXT NOT NULL,
                        source_kind TEXT NOT NULL,
                        case_kind TEXT NOT NULL,
                        selection_json TEXT NOT NULL,
                        seed INTEGER NOT NULL,
                        spec_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE source_runs (
                        id INTEGER PRIMARY KEY,
                        dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                        source_run_id INTEGER NOT NULL,
                        name TEXT NOT NULL DEFAULT '',
                        model_name TEXT NOT NULL DEFAULT '',
                        model_provider TEXT NOT NULL DEFAULT '',
                        config_json TEXT NOT NULL DEFAULT '{}',
                        pipeline_path TEXT,
                        usable_papers INTEGER NOT NULL DEFAULT 0,
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(dataset_id, source_run_id)
                    );
                    CREATE TABLE cases (
                        id INTEGER PRIMARY KEY,
                        dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                        case_key TEXT NOT NULL,
                        source_run_id INTEGER NOT NULL,
                        source_paper_id INTEGER NOT NULL,
                        source_trace_id INTEGER,
                        source_group_id TEXT NOT NULL,
                        paper_title TEXT NOT NULL DEFAULT '',
                        source_text TEXT NOT NULL,
                        text_kind TEXT NOT NULL,
                        pass1_text TEXT NOT NULL,
                        questions_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        original_answers_json TEXT NOT NULL DEFAULT '[]',
                        source_checksum TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(dataset_id, case_key)
                    );
                    CREATE INDEX cases_dataset_run ON cases(dataset_id, source_run_id);
                    CREATE TABLE model_configs (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        config_hash TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    );
                    CREATE UNIQUE INDEX model_configs_name ON model_configs(name);
                    CREATE TABLE executions (
                        id INTEGER PRIMARY KEY,
                        case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                        model_config_id INTEGER NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
                        status TEXT NOT NULL DEFAULT 'pending',
                        raw_response TEXT,
                        parsed_json TEXT,
                        diagnostics_json TEXT NOT NULL DEFAULT '{}',
                        usage_json TEXT NOT NULL DEFAULT '{}',
                        cost REAL,
                        latency_ms INTEGER,
                        error TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        UNIQUE(case_id, model_config_id)
                    );
                    CREATE INDEX executions_model_status ON executions(model_config_id, status);
                    CREATE TABLE answers (
                        id INTEGER PRIMARY KEY,
                        execution_id INTEGER NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
                        ordinal INTEGER NOT NULL,
                        key TEXT NOT NULL,
                        answer_json TEXT NOT NULL,
                        mechanical_status TEXT,
                        UNIQUE(execution_id, ordinal)
                    );
                    PRAGMA user_version=1;
                    """
                )
                version = 1
            if version < 2:
                # A lease makes interrupted processes recoverable while
                # retaining enough ownership information to avoid a stale
                # worker overwriting a newer worker's result.
                con.execute("ALTER TABLE executions ADD COLUMN run_token TEXT")
                con.execute("ALTER TABLE executions ADD COLUMN claimed_at TEXT")
                con.execute("UPDATE executions SET claimed_at=started_at WHERE status='running'")
                con.execute("CREATE INDEX executions_claimed_at ON executions(status, claimed_at)")
                con.execute("PRAGMA user_version=2")

    def create_dataset(self, spec: DatasetSpec) -> int:
        with self.transaction() as con:
            row = con.execute("SELECT * FROM datasets WHERE name=?", (spec.name,)).fetchone()
            if row:
                if row["spec_hash"] != spec.spec_hash:
                    raise ValueError(f"dataset {spec.name!r} already exists with a different specification")
                return int(row["id"])
            cur = con.execute(
                """INSERT INTO datasets
                (name,source_root,source_kind,case_kind,selection_json,seed,spec_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (spec.name, spec.source_root, spec.source_kind, spec.case_kind,
                 spec.selection_json, spec.seed, spec.spec_hash, _now()),
            )
            return int(cur.lastrowid)

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def add_source_run(self, dataset: str | int, *, source_run_id: int, name: str = "",
                       model_name: str = "", model_provider: str = "", config: Any = None,
                       pipeline_path: str | None = None, usable_papers: int = 0,
                       provenance: Any = None) -> int:
        dataset_id = self._dataset_id(dataset)
        with self.transaction() as con:
            con.execute(
                """INSERT INTO source_runs
                (dataset_id,source_run_id,name,model_name,model_provider,config_json,pipeline_path,usable_papers,provenance_json)
                VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(dataset_id,source_run_id) DO UPDATE SET
                name=excluded.name, model_name=excluded.model_name, model_provider=excluded.model_provider,
                config_json=excluded.config_json, pipeline_path=excluded.pipeline_path,
                usable_papers=excluded.usable_papers, provenance_json=excluded.provenance_json""",
                (dataset_id, source_run_id, name, model_name, model_provider,
                 canonical_json(redact_secrets(config or {})), pipeline_path, usable_papers,
                 canonical_json(redact_secrets(provenance or {}))),
            )
            return int(con.execute("SELECT id FROM source_runs WHERE dataset_id=? AND source_run_id=?",
                                   (dataset_id, source_run_id)).fetchone()[0])

    def add_case(self, case: BenchmarkCase) -> int:
        dataset_id = self._dataset_id(case.dataset)
        immutable = {
            "source_run_id": case.source_run_id, "source_paper_id": case.source_paper_id,
            "source_trace_id": case.source_trace_id, "source_group_id": case.source_group_id,
            "paper_title": case.paper_title, "source_text": case.source_text,
            "text_kind": case.text_kind, "pass1_text": case.pass1_text,
            "questions": redact_secrets(case.questions), "provenance": redact_secrets(dict(case.provenance)),
            "original_answers": redact_secrets(case.original_answers),
        }
        checksum = fingerprint(immutable)
        questions_json = canonical_json(immutable["questions"])
        provenance_json = canonical_json(immutable["provenance"])
        original_answers_json = canonical_json(immutable["original_answers"])
        with self.transaction() as con:
            existing = con.execute("SELECT * FROM cases WHERE dataset_id=? AND case_key=?",
                                   (dataset_id, case.case_key)).fetchone()
            if existing:
                # Frozen cases are an idempotency boundary. Never replace a
                # source/pass-1 snapshot after a model may have executed.
                comparable = {
                    "source_run_id": existing["source_run_id"], "source_paper_id": existing["source_paper_id"],
                    "source_trace_id": existing["source_trace_id"], "source_group_id": existing["source_group_id"],
                    "paper_title": existing["paper_title"], "source_text": existing["source_text"],
                    "text_kind": existing["text_kind"], "pass1_text": existing["pass1_text"],
                    "questions": json.loads(existing["questions_json"]),
                    "provenance": json.loads(existing["provenance_json"]),
                    "original_answers": json.loads(existing["original_answers_json"]),
                }
                if existing["source_checksum"] != checksum or comparable != immutable:
                    raise ValueError(f"benchmark case {case.case_key!r} already exists with different immutable content")
                return int(existing["id"])
            con.execute(
                """INSERT INTO cases
                (dataset_id,case_key,source_run_id,source_paper_id,source_trace_id,source_group_id,
                 paper_title,source_text,text_kind,pass1_text,questions_json,provenance_json,
                 original_answers_json,source_checksum,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (dataset_id, case.case_key, case.source_run_id, case.source_paper_id,
                 case.source_trace_id, case.source_group_id, case.paper_title, case.source_text,
                 case.text_kind, case.pass1_text, questions_json, provenance_json,
                 original_answers_json, checksum, _now()),
            )
            return int(con.execute("SELECT id FROM cases WHERE dataset_id=? AND case_key=?",
                                   (dataset_id, case.case_key)).fetchone()[0])

    def add_model_config(self, config: ModelConfig) -> int:
        with self.transaction() as con:
            row = con.execute("SELECT * FROM model_configs WHERE name=?", (config.name,)).fetchone()
            if row and row["config_hash"] != config.config_hash:
                raise ValueError(f"model config {config.name!r} already exists with a different fingerprint")
            if row:
                return int(row["id"])
            cur = con.execute("INSERT INTO model_configs(name,config_json,config_hash,created_at) VALUES (?,?,?,?)",
                              (config.name, canonical_json(redact_secrets(config.as_dict())), config.config_hash, _now()))
            return int(cur.lastrowid)

    def get_model_config(self, name_or_hash: str) -> dict[str, Any] | None:
        """Return a stored model configuration by name or fingerprint."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM model_configs WHERE name=? OR config_hash=?",
                (name_or_hash, name_or_hash),
            ).fetchone()
        return dict(row) if row else None

    def get_model_configs(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            return [dict(r) for r in con.execute("SELECT * FROM model_configs ORDER BY id").fetchall()]

    def ensure_executions(self, dataset: str, model_config: str | int) -> int:
        """Create pending execution rows for every case/config pair.

        ``INSERT OR IGNORE`` makes this safe to call on every resumable run.
        Returns the number of rows newly created.
        """
        dataset_id = self._dataset_id(dataset)
        config_id = self._model_config_id(model_config)
        with self.transaction() as con:
            before = int(con.execute("SELECT changes()").fetchone()[0])
            con.execute(
                """INSERT OR IGNORE INTO executions(case_id, model_config_id, status)
                   SELECT id, ?, 'pending' FROM cases WHERE dataset_id=?""",
                (config_id, dataset_id),
            )
            return int(con.execute("SELECT changes()").fetchone()[0]) - before

    def get_execution(self, case: str | int, model_config: str | int) -> dict[str, Any] | None:
        case_id = self._case_id(case)
        config_id = self._model_config_id(model_config)
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM executions WHERE case_id=? AND model_config_id=?",
                (case_id, config_id),
            ).fetchone()
        return dict(row) if row else None

    def get_executions(self, dataset: str, model_config: str | int | None = None) -> list[dict[str, Any]]:
        dataset_id = self._dataset_id(dataset)
        params: list[Any] = [dataset_id]
        query = """SELECT e.*, c.case_key, c.dataset_id, c.source_run_id, c.source_paper_id,
                          mc.name AS model_name, mc.config_hash
                   FROM executions e JOIN cases c ON c.id=e.case_id
                   JOIN model_configs mc ON mc.id=e.model_config_id
                   WHERE c.dataset_id=?"""
        if model_config is not None:
            query += " AND e.model_config_id=?"
            params.append(self._model_config_id(model_config))
        query += " ORDER BY c.source_run_id, c.source_paper_id, e.id"
        with self._connect() as con:
            return [dict(r) for r in con.execute(query, params).fetchall()]

    def recover_stale_running(self, stale_after_seconds: float = 3600.0) -> int:
        """Return abandoned running executions to pending.

        ``claimed_at`` is a lease timestamp. A long but active call should use
        a sufficiently conservative threshold; the runner default is one hour
        and callers/tests can choose a larger or smaller lease explicitly.
        """
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be non-negative")
        modifier = f"-{float(stale_after_seconds)} seconds"
        with self.transaction() as con:
            cur = con.execute(
                """UPDATE executions SET status='pending', run_token=NULL, claimed_at=NULL,
                   error='recovered abandoned running execution'
                   WHERE status='running' AND claimed_at IS NOT NULL
                   AND datetime(claimed_at) < datetime('now', ?)""",
                (modifier,),
            )
            return cur.rowcount

    def claim_execution(self, execution_id: int, *, retry_errors: bool = False,
                        run_token: str | None = None) -> bool:
        """Atomically transition a pending (or explicitly retried error) row to running."""
        token = run_token or uuid.uuid4().hex
        with self.transaction() as con:
            allowed = "status='pending'"
            if retry_errors:
                allowed += " OR status='error'"
            cur = con.execute(
                f"UPDATE executions SET status='running', started_at=?, finished_at=NULL, error=NULL, "
                f"run_token=?, claimed_at=? "
                f"WHERE id=? AND ({allowed})", (_now(), token, _now(), execution_id),
            )
            return cur.rowcount == 1

    def save_execution(
        self,
        execution_id: int,
        *,
        status: str,
        raw_response: str | None = None,
        parsed_answers: list[Mapping[str, Any]] | None = None,
        diagnostics: Any = None,
        usage: Any = None,
        cost: float | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
        run_token: str | None = None,
    ) -> None:
        """Commit an execution and all answer rows in one transaction."""
        if status not in {"pending", "running", "complete", "error"}:
            raise ValueError(f"unknown execution status {status!r}")
        answers = list(parsed_answers or [])
        with self.transaction() as con:
            where = "id=?"
            where_params: tuple[Any, ...] = (execution_id,)
            if run_token is not None:
                where = "id=? AND run_token=?"
                where_params = (execution_id, run_token)
            cur = con.execute(
                """UPDATE executions SET status=?, raw_response=?, parsed_json=?, diagnostics_json=?,
                   usage_json=?, cost=?, latency_ms=?, error=?, finished_at=?, run_token=NULL,
                   claimed_at=NULL WHERE """ + where,
                (status, raw_response,
                 canonical_json(answers), canonical_json(redact_secrets(diagnostics or {})),
                 # Usage keys such as ``input_tokens`` are metrics, not
                 # credentials; do not pass them through the broad source
                 # provenance redactor (which intentionally catches token
                 # fields).
                 canonical_json(usage or {}), cost, latency_ms, error,
                 _now() if status in {"complete", "error"} else None, *where_params),
            )
            if cur.rowcount != 1:
                raise RuntimeError("execution lease was lost before result could be saved")
            if status in {"complete", "error"}:
                con.execute("DELETE FROM answers WHERE execution_id=?", (execution_id,))
                for ordinal, answer in enumerate(answers):
                    con.execute(
                        """INSERT INTO answers(execution_id, ordinal, key, answer_json, mechanical_status)
                           VALUES (?,?,?,?,?)""",
                        (execution_id, ordinal, str(answer.get("key", "")), canonical_json(redact_secrets(dict(answer))),
                         answer.get("status")),
                    )

    def get_answers(self, execution_id: int) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM answers WHERE execution_id=? ORDER BY ordinal", (execution_id,)
            ).fetchall()
        return [dict(row) | {"answer": json.loads(row["answer_json"])} for row in rows]

    def _model_config_id(self, model_config: str | int) -> int:
        if isinstance(model_config, int):
            return model_config
        with self._connect() as con:
            row = con.execute("SELECT id FROM model_configs WHERE name=? OR config_hash=?",
                              (model_config, model_config)).fetchone()
        if not row:
            raise KeyError(f"unknown benchmark model config {model_config!r}")
        return int(row[0])

    def _case_id(self, case: str | int) -> int:
        if isinstance(case, int):
            return case
        with self._connect() as con:
            row = con.execute("SELECT id FROM cases WHERE case_key=?", (case,)).fetchone()
        if not row:
            raise KeyError(f"unknown benchmark case {case!r}")
        return int(row[0])

    def get_cases(self, dataset: str, *, source_run_ids: list[int] | None = None) -> list[dict[str, Any]]:
        dataset_id = self._dataset_id(dataset)
        query = "SELECT * FROM cases WHERE dataset_id=?"
        params: list[Any] = [dataset_id]
        if source_run_ids:
            query += " AND source_run_id IN (" + ",".join("?" for _ in source_run_ids) + ")"
            params.extend(source_run_ids)
        query += " ORDER BY source_run_id, source_paper_id, id"
        with self._connect() as con:
            return [dict(r) for r in con.execute(query, params).fetchall()]

    def get_cases_by_id(self, case_id: int) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return dict(row) if row else None

    def _dataset_id(self, dataset: str | int) -> int:
        if isinstance(dataset, int):
            return dataset
        with self._connect() as con:
            row = con.execute("SELECT id FROM datasets WHERE name=?", (dataset,)).fetchone()
        if not row:
            raise KeyError(f"unknown benchmark dataset {dataset!r}")
        return int(row[0])
