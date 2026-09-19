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
                    paper_id   INTEGER NOT NULL,
                    variant    TEXT NOT NULL DEFAULT '',
                    markdown   TEXT,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, variant)
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
                    posted_at       TEXT,
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
                    posted_at       TEXT,
                    PRIMARY KEY (arbiter_run_id, dispute_item_id)
                );
            """)
            self._add_posted_at(con, "answers")
            self._add_posted_at(con, "resolutions")
            self._add_ocr_variant(con)

    @staticmethod
    def _add_posted_at(con: sqlite3.Connection, table: str) -> None:
        """Add the `posted_at` column to a store file written before it existed.

        Job stores are per-job files that outlive the code that wrote them (a batch
        job is resumed days later, from a `state.db` on disk), so the schema has to
        be brought forward in place. Rows already at status 'posted' were delivered,
        so they are backfilled — otherwise a resume would post every one of them a
        second time.
        """
        cols = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
        if "posted_at" in cols:
            return
        con.execute(f"ALTER TABLE {table} ADD COLUMN posted_at TEXT")
        con.execute(f"UPDATE {table} SET posted_at = updated_at WHERE status = 'posted'")

    @staticmethod
    def _add_ocr_variant(con: sqlite3.Connection) -> None:
        """Re-key an `ocr_cache` written before the cache knew about renderings.

        A store file outlives the code that wrote it (a batch job is resumed days
        later from a `state.db` on disk), so the schema is brought forward in
        place — see `_add_posted_at`. SQLite cannot add a column to a primary
        key, so the table is rebuilt rather than altered.

        Carried-over rows land on the variant ``'legacy'``, which no client ever
        asks for. That is deliberate: the old schema recorded the text but not
        which rendering it was, so serving such a row to a run would be a guess.
        Keeping them costs nothing (one refetch each, once) and leaves the
        resumed job's text visible to `get_ocr_any`.
        """
        cols = {row["name"] for row in con.execute("PRAGMA table_info(ocr_cache)")}
        if not cols or "variant" in cols:
            return
        con.execute("ALTER TABLE ocr_cache RENAME TO ocr_cache_pre_variant")
        con.execute(
            """CREATE TABLE ocr_cache (
                   paper_id   INTEGER NOT NULL,
                   variant    TEXT NOT NULL DEFAULT '',
                   markdown   TEXT,
                   fetched_at TEXT NOT NULL,
                   PRIMARY KEY (paper_id, variant)
               )"""
        )
        con.execute(
            """INSERT INTO ocr_cache (paper_id, variant, markdown, fetched_at)
               SELECT paper_id, 'legacy', markdown, fetched_at FROM ocr_cache_pre_variant"""
        )
        con.execute("DROP TABLE ocr_cache_pre_variant")

    # ------------------------------------------------------------------
    # OCR cache
    # ------------------------------------------------------------------

    def get_ocr(self, paper_id: int, variant: str) -> str | None:
        """Return the cached text of one *rendering* of a paper, or None.

        `variant` is the opaque key the client gave for the run being served
        (`SeerClient.ocr_variant`). One paper has as many cache rows as the runs
        in this job have distinct renderings of it — a run that withholds the
        reference list and one that does not are two different texts, and the
        paper id alone cannot tell them apart.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT markdown FROM ocr_cache WHERE paper_id = ? AND variant = ?",
                (paper_id, variant),
            ).fetchone()
        return row["markdown"] if row else None

    def get_ocr_any(self, paper_id: int) -> str | None:
        """Return the most recently fetched rendering of a paper, whichever it is.

        For readers with no run in hand — the debug UI. Never for assembling a
        prompt or verifying a quote: those must ask for the rendering their own
        run was sent, via `get_ocr`.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT markdown FROM ocr_cache WHERE paper_id = ? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (paper_id,),
            ).fetchone()
        return row["markdown"] if row else None

    def save_ocr(self, paper_id: int, markdown: str | None, variant: str) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO ocr_cache (paper_id, variant, markdown, fetched_at)"
                " VALUES (?,?,?,?)",
                (paper_id, variant, markdown, _now()),
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
        """Save a finished cell's payload as undelivered.

        INSERT OR REPLACE writes a whole new row, so a retry's payload arrives with
        `posted_at` back at NULL — which is what makes the retry's answer replace
        the error answer that was posted for the earlier attempt.
        """
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
        """Mark the cell for recomputation on the next run. Says nothing about
        delivery: the error answer this row holds is still owed to SEER, and
        `posted_at` — deliberately untouched here — is what tracks that."""
        with self._tx() as con:
            con.execute(
                """UPDATE answers SET status='failed', error=?, updated_at=?
                   WHERE run_id=? AND paper_id=? AND version_id=?""",
                (error, _now(), run_id, paper_id, version_id),
            )

    def mark_posted(self, run_id: int, paper_id: int, version_ids: list[int]) -> None:
        """Record that these payloads reached SEER.

        Only a 'done' row becomes 'posted'. A 'failed' row keeps its status, because
        status decides whether the cell is recomputed next run and delivering its
        error answer is no reason to stop retrying it.
        """
        now = _now()
        with self._tx() as con:
            con.executemany(
                """UPDATE answers
                      SET posted_at=?, updated_at=?,
                          status=CASE WHEN status='done' THEN 'posted' ELSE status END
                    WHERE run_id=? AND paper_id=? AND version_id=?""",
                [(now, now, run_id, paper_id, vid) for vid in version_ids],
            )

    def get_unposted(self, run_id: int, paper_id: int) -> list[dict]:
        """Payloads saved for this paper that SEER has not been sent yet.

        Keyed on `posted_at`, never on status, because the two answer different
        questions — has this been delivered, and should this cell be recomputed —
        and a dropped cell is 'yes' to both. Selecting on status='done' alone is
        what used to make `mark_failed` silently cancel the post of an error
        answer, so the paper vanished from a run with no error to show for it.

        'pass1_done' rows carry a payload too, but a half-processed cell is not an
        answer, so the statuses that can be posted are listed explicitly.
        """
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND posted_at IS NULL
                   AND status IN ('done', 'failed')
                   AND payload_json IS NOT NULL""",
                (run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_postable(self, run_id: int, paper_id: int) -> list[dict]:
        """Return every payload this store holds for the paper, delivered or not
        (for repost). Includes 'failed' rows: an error answer is an answer, and a
        repost that skipped them would rebuild the same silent gap."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM answers
                   WHERE run_id=? AND paper_id=? AND status IN ('done', 'posted', 'failed')
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
        """The resolutions twin of `mark_failed` — retryability only, never delivery."""
        with self._tx() as con:
            con.execute(
                """UPDATE resolutions SET status='failed', error=?, updated_at=?
                   WHERE arbiter_run_id=? AND dispute_item_id=?""",
                (error, _now(), arbiter_run_id, dispute_item_id),
            )

    def mark_resolutions_posted(self, arbiter_run_id: int, dispute_item_ids: list[int]) -> None:
        """The resolutions twin of `mark_posted` — see it for why a 'failed' row
        keeps its status."""
        now = _now()
        with self._tx() as con:
            con.executemany(
                """UPDATE resolutions
                      SET posted_at=?, updated_at=?,
                          status=CASE WHEN status='done' THEN 'posted' ELSE status END
                    WHERE arbiter_run_id=? AND dispute_item_id=?""",
                [(now, now, arbiter_run_id, did) for did in dispute_item_ids],
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
        """The resolutions twin of `get_unposted` — see it for why this selects on
        `posted_at` rather than status."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND posted_at IS NULL
                   AND status IN ('done', 'failed')
                   AND payload_json IS NOT NULL""",
                (arbiter_run_id, paper_id),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_postable_resolutions(self, arbiter_run_id: int, paper_id: int) -> list[dict]:
        """Return every resolution payload this store holds for the paper, delivered
        or not (for repost). Includes 'failed' rows, like `get_postable`."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT payload_json FROM resolutions
                   WHERE arbiter_run_id=? AND paper_id=? AND status IN ('done', 'posted', 'failed')
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

    def owes_batch(self, batch_id: str) -> bool:
        """Is `batch_id` still owed — submitted, and not yet collected?

        `submit_and_poll` saves the id before polling and deletes it once
        `collect()` has returned (or the batch came back failed), so presence
        here is the one durable record of "this batch still has results nobody
        has taken". A caller that tracks batches in its own database has no
        other way to tell a batch it is still waiting on from one that was
        collected hours ago: the provider will happily serve the results again,
        and the job row alone does not say whether anyone did.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM kv WHERE value=? LIMIT 1", (batch_id,),
            ).fetchone()
        return row is not None

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
