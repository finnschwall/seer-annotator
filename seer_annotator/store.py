"""SQLite-backed local state: OCR cache + answer idempotency."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator

ANSWER_STATUSES = {"pending", "done", "posted", "failed", "skipped", "pass1_done"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_is_error(payload_json: str | None) -> bool:
    """Apply `mapping.payload_is_error` to a stored payload, tolerating junk.

    Imported lazily: `mapping` is a leaf module today, and keeping the import
    inside the function means `store` stays importable even if that ever changes.
    """
    from .mapping import payload_is_error

    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload_is_error(payload)


class Store:
    def __init__(self, path: str = ".seer_state.db") -> None:
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
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
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    paper_id   INTEGER PRIMARY KEY,
                    markdown   TEXT,
                    fetched_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS answers (
                    run_id          INTEGER NOT NULL,
                    paper_id        INTEGER NOT NULL,
                    version_id      INTEGER NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    payload_json    TEXT,
                    batch_group_id  TEXT,
                    error           TEXT,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY (run_id, paper_id, version_id)
                );

                CREATE TABLE IF NOT EXISTS kv (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resolutions (
                    arbiter_run_id  INTEGER NOT NULL,
                    dispute_item_id INTEGER NOT NULL,
                    paper_id        INTEGER NOT NULL,
                    version_id      INTEGER NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    payload_json    TEXT,
                    batch_group_id  TEXT,
                    error           TEXT,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY (arbiter_run_id, dispute_item_id)
                );
            """)

    # ------------------------------------------------------------------
    # OCR cache
    # ------------------------------------------------------------------

    def get_ocr(self, paper_id: int) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT markdown FROM ocr_cache WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        return row["markdown"] if row else None

    def save_ocr(self, paper_id: int, markdown: str | None) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO ocr_cache (paper_id, markdown, fetched_at) VALUES (?,?,?)",
                (paper_id, markdown, _now()),
            )

    # ------------------------------------------------------------------
    # Answer state
    # ------------------------------------------------------------------

    def get_status(self, run_id: int, paper_id: int, version_id: int) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM answers WHERE run_id=? AND paper_id=? AND version_id=?",
                (run_id, paper_id, version_id),
            ).fetchone()
        return row["status"] if row else None

    def upsert_pending(self, run_id: int, paper_id: int, version_id: int) -> None:
        """Insert as pending only if no row exists yet."""
        with self._tx() as con:
            con.execute(
                """INSERT OR IGNORE INTO answers
                   (run_id, paper_id, version_id, status, updated_at)
                   VALUES (?,?,?,'pending',?)""",
                (run_id, paper_id, version_id, _now()),
            )

    def save_answer(
        self,
        run_id: int,
        paper_id: int,
        version_id: int,
        payload: dict,
        batch_group_id: str | None = None,
    ) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO answers
                   (run_id, paper_id, version_id, status, payload_json, batch_group_id, updated_at)
                   VALUES (?,?,?,'done',?,?,?)""",
                (
                    run_id,
                    paper_id,
                    version_id,
                    json.dumps(payload),
                    batch_group_id,
                    _now(),
                ),
            )

    def save_pass1(
        self,
        run_id: int,
        paper_id: int,
        version_id: int,
        payload: dict,
        batch_group_id: str | None = None,
    ) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO answers
                   (run_id, paper_id, version_id, status, payload_json, batch_group_id, updated_at)
                   VALUES (?,?,?,'pass1_done',?,?,?)""",
                (
                    run_id,
                    paper_id,
                    version_id,
                    json.dumps(payload),
                    batch_group_id,
                    _now(),
                ),
            )

    def mark_skipped(self, run_id: int, paper_id: int, version_id: int, reason: str = "") -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO answers
                   (run_id, paper_id, version_id, status, error, updated_at)
                   VALUES (?,?,?,'skipped',?,?)""",
                (run_id, paper_id, version_id, reason, _now()),
            )

    def mark_failed(self, run_id: int, paper_id: int, version_id: int, error: str) -> None:
        with self._tx() as con:
            con.execute(
                """UPDATE answers SET status='failed', error=?, updated_at=?
                   WHERE run_id=? AND paper_id=? AND version_id=?""",
                (error, _now(), run_id, paper_id, version_id),
            )

    def mark_posted(self, run_id: int, paper_id: int, version_ids: list[int]) -> None:
        with self._tx() as con:
            con.executemany(
                """UPDATE answers SET status='posted', updated_at=?
                   WHERE run_id=? AND paper_id=? AND version_id=?""",
                [(_now(), run_id, paper_id, vid) for vid in version_ids],
            )

    def get_unposted(self, run_id: int, paper_id: int) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND status='done'
                   AND payload_json IS NOT NULL""",
                (run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_postable(self, run_id: int, paper_id: int) -> list[dict]:
        """Return payloads for answers that are done or already posted (for repost)."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND status IN ('done', 'posted')
                   AND payload_json IS NOT NULL""",
                (run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def should_skip_cell(self, run_id: int, paper_id: int, version_id: int) -> bool:
        """Return True if this cell is already done/posted and should not be recomputed."""
        status = self.get_status(run_id, paper_id, version_id)
        return status in ("done", "posted")

    def finished_cells(
        self, run_id: int, paper_ids: Iterable[int], version_ids: Iterable[int]
    ) -> dict:
        """Return ``{(paper_id, version_id): is_error}`` for the already-finished cells
        of ``run_id`` — what the orchestrator's per-cell counter has counted so far.

        A run that submits provider batches does not live in one process: every
        ``BatchPendingError`` parks the job, and ``run_pipeline`` is later called again
        from the top. Cells finished by an earlier invocation are dropped before they
        can be counted (``should_skip_cell`` in ``batch_runner``), so a counter starting
        at 0 reports only the last invocation's own share of the work — and SEER
        downgrades a 'succeeded' heartbeat short of ``cells_total`` to 'failed'.

        Returned per cell rather than as a total because a resume also *re*-processes
        every cell that is not `done`/`posted` — a failed cell, and the finished cells
        of its question group with it — and each of those must still count exactly once
        when it comes round again.

        "Finished" means a payload was saved, which is exactly the set of cells that
        went through ``_count_payload``. Rows left at ``pass1_done`` carry a payload too
        but are only half processed, so they are excluded. Restricted to the papers and
        question versions of the caller's current scope, so the result can never exceed
        that scope's ``cells_total``.
        """
        papers = set(paper_ids)
        versions = set(version_ids)
        out: dict = {}
        with self._connect() as con:
            for row in con.execute(
                """SELECT paper_id, version_id, payload_json FROM answers
                   WHERE run_id=? AND payload_json IS NOT NULL AND status<>'pass1_done'""",
                (run_id,),
            ):
                if row["paper_id"] not in papers or row["version_id"] not in versions:
                    continue
                out[(row["paper_id"], row["version_id"])] = _payload_is_error(row["payload_json"])
        return out

    def reset_runs(self, run_ids: list[int]) -> None:
        """Delete all cached answer rows for these run ids, forcing a fresh re-annotation on
        the next run without disturbing other runs' cached state in a shared store."""
        if not run_ids:
            return
        with self._tx() as con:
            placeholders = ",".join("?" * len(run_ids))
            con.execute(f"DELETE FROM answers WHERE run_id IN ({placeholders})", run_ids)

    # ------------------------------------------------------------------
    # Resolution state (arbitration) — mirrors the answers-table methods above,
    # keyed by (arbiter_run_id, dispute_item_id) instead of (run_id, version_id).
    # ------------------------------------------------------------------

    def get_resolution_status(self, arbiter_run_id: int, dispute_item_id: int) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM resolutions WHERE arbiter_run_id=? AND dispute_item_id=?",
                (arbiter_run_id, dispute_item_id),
            ).fetchone()
        return row["status"] if row else None

    def upsert_pending_resolution(
        self, arbiter_run_id: int, dispute_item_id: int, paper_id: int, version_id: int
    ) -> None:
        """Insert as pending only if no row exists yet."""
        with self._tx() as con:
            con.execute(
                """INSERT OR IGNORE INTO resolutions
                   (arbiter_run_id, dispute_item_id, paper_id, version_id, status, updated_at)
                   VALUES (?,?,?,?,'pending',?)""",
                (arbiter_run_id, dispute_item_id, paper_id, version_id, _now()),
            )

    def save_resolution(
        self,
        arbiter_run_id: int,
        dispute_item_id: int,
        paper_id: int,
        version_id: int,
        payload: dict,
        batch_group_id: str | None = None,
    ) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO resolutions
                   (arbiter_run_id, dispute_item_id, paper_id, version_id, status, payload_json, batch_group_id, updated_at)
                   VALUES (?,?,?,?,'done',?,?,?)""",
                (
                    arbiter_run_id,
                    dispute_item_id,
                    paper_id,
                    version_id,
                    json.dumps(payload),
                    batch_group_id,
                    _now(),
                ),
            )

    def save_pass1_resolution(
        self,
        arbiter_run_id: int,
        dispute_item_id: int,
        paper_id: int,
        version_id: int,
        payload: dict,
        batch_group_id: str | None = None,
    ) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO resolutions
                   (arbiter_run_id, dispute_item_id, paper_id, version_id, status, payload_json, batch_group_id, updated_at)
                   VALUES (?,?,?,?,'pass1_done',?,?,?)""",
                (
                    arbiter_run_id,
                    dispute_item_id,
                    paper_id,
                    version_id,
                    json.dumps(payload),
                    batch_group_id,
                    _now(),
                ),
            )

    def mark_resolution_skipped(
        self, arbiter_run_id: int, dispute_item_id: int, paper_id: int, version_id: int, reason: str = ""
    ) -> None:
        with self._tx() as con:
            con.execute(
                """INSERT OR REPLACE INTO resolutions
                   (arbiter_run_id, dispute_item_id, paper_id, version_id, status, error, updated_at)
                   VALUES (?,?,?,?,'skipped',?,?)""",
                (arbiter_run_id, dispute_item_id, paper_id, version_id, reason, _now()),
            )

    def mark_resolution_failed(self, arbiter_run_id: int, dispute_item_id: int, error: str) -> None:
        with self._tx() as con:
            con.execute(
                """UPDATE resolutions SET status='failed', error=?, updated_at=?
                   WHERE arbiter_run_id=? AND dispute_item_id=?""",
                (error, _now(), arbiter_run_id, dispute_item_id),
            )

    def mark_resolutions_posted(self, arbiter_run_id: int, dispute_item_ids: list[int]) -> None:
        with self._tx() as con:
            con.executemany(
                """UPDATE resolutions SET status='posted', updated_at=?
                   WHERE arbiter_run_id=? AND dispute_item_id=?""",
                [(_now(), arbiter_run_id, did) for did in dispute_item_ids],
            )

    def finished_resolutions(self, arbiter_run_id: int, dispute_item_ids: Iterable[int]) -> dict:
        """Return ``{dispute_item_id: is_error}`` for the already-finished dispute items
        of ``arbiter_run_id`` — the resolutions-table twin of ``finished_cells``, and
        needed for the same reason (see that method). Arbitration's cell is one dispute
        item, so the scope filter is a set of dispute item ids rather than papers x
        question versions.
        """
        items = set(dispute_item_ids)
        out: dict = {}
        with self._connect() as con:
            for row in con.execute(
                """SELECT dispute_item_id, payload_json FROM resolutions
                   WHERE arbiter_run_id=? AND payload_json IS NOT NULL
                     AND status<>'pass1_done'""",
                (arbiter_run_id,),
            ):
                if row["dispute_item_id"] not in items:
                    continue
                out[row["dispute_item_id"]] = _payload_is_error(row["payload_json"])
        return out

    def reset_arbiter_runs(self, arbiter_run_ids: list[int]) -> None:
        """Delete all cached resolution rows for these arbiter run ids, forcing fresh
        re-adjudication on the next run without disturbing other runs' cached state in
        a shared store. Mirrors reset_runs above, but targets the resolutions table
        (keyed by arbiter_run_id) since reset_runs is hardcoded to the answers table."""
        if not arbiter_run_ids:
            return
        with self._tx() as con:
            placeholders = ",".join("?" * len(arbiter_run_ids))
            con.execute(
                f"DELETE FROM resolutions WHERE arbiter_run_id IN ({placeholders})", arbiter_run_ids
            )

    def get_unposted_resolutions(self, arbiter_run_id: int, paper_id: int) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND status='done'
                   AND payload_json IS NOT NULL""",
                (arbiter_run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_postable_resolutions(self, arbiter_run_id: int, paper_id: int) -> list[dict]:
        """Return payloads for resolutions that are done or already posted (for repost)."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND status IN ('done', 'posted')
                   AND payload_json IS NOT NULL""",
                (arbiter_run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def should_skip_resolution_cell(self, arbiter_run_id: int, dispute_item_id: int) -> bool:
        """Return True if this dispute is already done/posted and should not be recomputed."""
        status = self.get_resolution_status(arbiter_run_id, dispute_item_id)
        return status in ("done", "posted")

    def should_skip_resolution_cell_by_paper_version(
        self, arbiter_run_id: int, paper_id: int, version_id: int
    ) -> bool:
        """Like should_skip_resolution_cell, keyed by (paper_id, version_id) instead of
        dispute_item_id — for use as batch_runner's should_skip_cell callback, which
        only knows (run_id, paper_id, version_id). A dispute is unique per
        (dispute_set, paper, question_version), so this triple is unambiguous."""
        with self._connect() as con:
            row = con.execute(
                """SELECT status FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND version_id=?""",
                (arbiter_run_id, paper_id, version_id),
            ).fetchone()
        return (row["status"] if row else None) in ("done", "posted")

    def get_pass1_resolution_rows(self, arbiter_run_id: int, paper_id: int) -> list[dict]:
        """Return pass1_done resolution rows with parsed payloads, ready for Pass-2 processing."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT arbiter_run_id, dispute_item_id, paper_id, version_id, status, payload_json
                   FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND status='pass1_done'
                   AND payload_json IS NOT NULL""",
                (arbiter_run_id, paper_id),
            ).fetchall()
        return [
            {
                "arbiter_run_id": row["arbiter_run_id"],
                "dispute_item_id": row["dispute_item_id"],
                "paper_id": row["paper_id"],
                "version_id": row["version_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_reformattable_resolution_rows(self, arbiter_run_id: int, paper_id: int) -> list[dict]:
        """Return done/posted resolution rows with parsed payloads, ready for reformatting."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT arbiter_run_id, dispute_item_id, paper_id, version_id, status, payload_json
                   FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND status IN ('done', 'posted')
                   AND payload_json IS NOT NULL""",
                (arbiter_run_id, paper_id),
            ).fetchall()
        return [
            {
                "arbiter_run_id": row["arbiter_run_id"],
                "dispute_item_id": row["dispute_item_id"],
                "paper_id": row["paper_id"],
                "version_id": row["version_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def update_reformatted_resolution(
        self, arbiter_run_id: int, dispute_item_id: int, payload: dict
    ) -> None:
        """Replace stored payload after a reformat and reset status to done."""
        with self._tx() as con:
            con.execute(
                """UPDATE resolutions SET payload_json=?, status='done', updated_at=?
                   WHERE arbiter_run_id=? AND dispute_item_id=?""",
                (json.dumps(payload), _now(), arbiter_run_id, dispute_item_id),
            )

    def resolution_stats(self) -> dict:
        with self._connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) as n FROM resolutions GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def resolution_cost_summary(self) -> list[dict]:
        """Aggregate token/cost totals per arbiter run from stored resolution payloads."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT arbiter_run_id, payload_json FROM resolutions WHERE payload_json IS NOT NULL"
            ).fetchall()

        totals: dict[int, dict] = {}
        for row in rows:
            rid = row["arbiter_run_id"]
            payload = json.loads(row["payload_json"])
            t = totals.setdefault(
                rid,
                {
                    "run_id": rid,
                    "tokens_total": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "tokens_cached": 0,
                    "cost_usd": 0.0,
                    "fmt_tokens_total": 0,
                    "fmt_tokens_input": 0,
                    "fmt_tokens_output": 0,
                    "fmt_tokens_cached": 0,
                    "fmt_cost_usd": 0.0,
                    "answers": 0,
                },
            )
            t["tokens_total"] += payload.get("tokens_total", 0) or 0
            t["tokens_input"] += payload.get("tokens_input", 0) or 0
            t["tokens_output"] += payload.get("tokens_output", 0) or 0
            t["tokens_cached"] += payload.get("tokens_cached", 0) or 0
            cost = payload.get("cost")
            if cost:
                t["cost_usd"] += float(cost)
            t["fmt_tokens_total"] += payload.get("fmt_tokens_total", 0) or 0
            t["fmt_tokens_input"] += payload.get("fmt_tokens_input", 0) or 0
            t["fmt_tokens_output"] += payload.get("fmt_tokens_output", 0) or 0
            t["fmt_tokens_cached"] += payload.get("fmt_tokens_cached", 0) or 0
            fmt_cost = payload.get("fmt_cost")
            if fmt_cost:
                t["fmt_cost_usd"] += float(fmt_cost)
            t["answers"] += 1

        return list(totals.values())

    # ------------------------------------------------------------------
    # Batch job ID persistence (for resuming async batch runs)
    # ------------------------------------------------------------------

    def save_batch_id(self, key: str, batch_id: str) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)",
                (key, batch_id),
            )

    def get_batch_id(self, key: str) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def delete_batch_id(self, key: str) -> None:
        with self._tx() as con:
            con.execute("DELETE FROM kv WHERE key=?", (key,))

    # ------------------------------------------------------------------
    # UI / inspection helpers
    # ------------------------------------------------------------------

    def all_answers(
        self,
        run_id: int | None = None,
        paper_id: int | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM answers WHERE 1=1"
        params: list = []
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        if paper_id is not None:
            query += " AND paper_id=?"
            params.append(paper_id)
        query += " ORDER BY run_id, paper_id, version_id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) as n FROM answers GROUP BY status"
            ).fetchall()
            ocr_count = con.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()[0]
        return {
            "ocr_cached": ocr_count,
            "answers": {r["status"]: r["n"] for r in rows},
        }

    def get_reformattable_rows(self, run_id: int, paper_id: int) -> list[dict]:
        """Return done/posted answer rows with parsed payloads, ready for reformatting."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT run_id, paper_id, version_id, status, payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND status IN ('done', 'posted')
                   AND payload_json IS NOT NULL""",
                (run_id, paper_id),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "paper_id": row["paper_id"],
                "version_id": row["version_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_pass1_rows(self, run_id: int, paper_id: int) -> list[dict]:
        """Return pass1_done answer rows with parsed payloads, ready for Pass-2 processing."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT run_id, paper_id, version_id, status, payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND status='pass1_done'
                   AND payload_json IS NOT NULL""",
                (run_id, paper_id),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "paper_id": row["paper_id"],
                "version_id": row["version_id"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def update_reformatted(
        self, run_id: int, paper_id: int, version_id: int, payload: dict
    ) -> None:
        """Replace stored payload after a reformat and reset status to done."""
        with self._tx() as con:
            con.execute(
                """UPDATE answers SET payload_json=?, status='done', updated_at=?
                   WHERE run_id=? AND paper_id=? AND version_id=?""",
                (json.dumps(payload), _now(), run_id, paper_id, version_id),
            )

    def cost_summary(self) -> list[dict]:
        """Aggregate token/cost totals per run from stored payloads."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT run_id, payload_json FROM answers WHERE payload_json IS NOT NULL"
            ).fetchall()

        totals: dict[int, dict] = {}
        for row in rows:
            rid = row["run_id"]
            payload = json.loads(row["payload_json"])
            t = totals.setdefault(
                rid,
                {
                    "run_id": rid,
                    "tokens_total": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "tokens_cached": 0,
                    "cost_usd": 0.0,
                    "fmt_tokens_total": 0,
                    "fmt_tokens_input": 0,
                    "fmt_tokens_output": 0,
                    "fmt_tokens_cached": 0,
                    "fmt_cost_usd": 0.0,
                    "answers": 0,
                },
            )
            t["tokens_total"] += payload.get("tokens_total", 0) or 0
            t["tokens_input"] += payload.get("tokens_input", 0) or 0
            t["tokens_output"] += payload.get("tokens_output", 0) or 0
            t["tokens_cached"] += payload.get("tokens_cached", 0) or 0
            cost = payload.get("cost")
            if cost:
                t["cost_usd"] += float(cost)
            t["fmt_tokens_total"] += payload.get("fmt_tokens_total", 0) or 0
            t["fmt_tokens_input"] += payload.get("fmt_tokens_input", 0) or 0
            t["fmt_tokens_output"] += payload.get("fmt_tokens_output", 0) or 0
            t["fmt_tokens_cached"] += payload.get("fmt_tokens_cached", 0) or 0
            fmt_cost = payload.get("fmt_cost")
            if fmt_cost:
                t["fmt_cost_usd"] += float(fmt_cost)
            t["answers"] += 1

        return list(totals.values())
