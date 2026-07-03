"""Heartbeat client for SEER-orchestrated runs.

When SEER launches `seer-annotate run` as a background subprocess it passes
--progress-url, and expects heartbeat POSTs at run start, after each chunk, and
on completion. See README.md "Progress / heartbeat contract" for the wire shape —
this module is a thin, best-effort client over that contract: a broken or
unreachable progress endpoint must never fail the run itself.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


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
    ) -> None:
        """POST a heartbeat. No-ops when no progress_url was configured.

        Best-effort: logs and swallows any failure rather than raising, since a
        broken progress callback must never abort the underlying run.
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
        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=10) as client:
                resp = await client.post(self._url, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Progress heartbeat failed (run=%s, status=%s): %s", self._run_id, status, exc)
