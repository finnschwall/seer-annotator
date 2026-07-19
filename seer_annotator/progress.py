"""Heartbeat client for SEER-orchestrated runs.

When SEER launches `seer-annotate run` as a background subprocess it passes
--progress-url, and expects heartbeat POSTs at run start, after each chunk, and
on completion. See README.md "Progress / heartbeat contract" for the wire shape —
this module is a thin, best-effort client over that contract: a broken or
unreachable progress endpoint must never fail the run itself.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class ProgressReporterProtocol(Protocol):
    """Structural interface satisfied by ProgressReporter (and any in-process
    stand-in an embedder wants to inject via ``reporter_factory``) — same keyword
    signature as ``ProgressReporter.heartbeat``, so orchestrator code can call
    either without caring which one it got."""

    async def heartbeat(
        self,
        *,
        status: str,
        cells_total: int,
        cells_done: int,
        cells_error: int,
        message: str = "",
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        phase: str | None = None,
        phase_done: int | None = None,
        phase_total: int | None = None,
    ) -> None: ...


class ProgressReporter:
    def __init__(self, progress_url: str | None, api_token: str, run_id: int) -> None:
        self._url = progress_url
        self._run_id = run_id
        self._headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    async def heartbeat(
        self,
        *,
        status: str,
        cells_total: int,
        cells_done: int,
        cells_error: int,
        message: str = "",
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        phase: str | None = None,
        phase_done: int | None = None,
        phase_total: int | None = None,
    ) -> None:
        """POST a heartbeat. No-ops when no progress_url was configured.

        Best-effort: logs and swallows any failure rather than raising, since a
        broken progress callback must never abort the underlying run.

        ``chunk_index``/``chunk_total``/``phase``/``phase_done``/``phase_total``
        are optional fields that convey which chunk and which pass
        (``"pass1"``/``"pass2"``) is currently in flight, and progress within
        it. Guaranteed heartbeats (run start, chunk boundary, terminal) leave
        these ``None`` — that signals "no phase in flight right now" rather
        than "unknown," so the server can tell the difference between a
        mid-chunk update and one that should leave prior phase state alone
        (SEER's progress endpoint clears phase fields only on terminal status).
        A ``None`` here is omitted from the POSTed JSON entirely rather than
        sent as a literal ``null`` — the receiver treats "key absent" as
        "no update," but treats a present ``null`` the same as any other
        value, so sending an explicit ``null`` would clobber the last known
        phase/chunk state instead of leaving it alone.
        """
        if not self._url:
            return

        payload = {
            "run_id": self._run_id,
            "status": status,
            "cells_total": cells_total,
            "cells_done": cells_done,
            "cells_error": cells_error,
            "message": message,
        }
        # Omit rather than null these out when not in flight — the receiver
        # distinguishes "no update to this dimension" (key absent) from an
        # explicit clear, and treats a present `null` the same as any other
        # value. Sending `null` here would actively clobber the last known
        # phase/chunk state on every guaranteed heartbeat instead of leaving
        # it alone as intended.
        for key, value in (
            ("chunk_index", chunk_index),
            ("chunk_total", chunk_total),
            ("phase", phase),
            ("phase_done", phase_done),
            ("phase_total", phase_total),
        ):
            if value is not None:
                payload[key] = value
        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=10) as client:
                resp = await client.post(self._url, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Progress heartbeat failed (run=%s, status=%s): %s", self._run_id, status, exc)
