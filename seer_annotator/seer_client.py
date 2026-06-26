"""SEER API client: fetch OCR markdown, post LLMAnswers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .config import Question

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5


def _coerce_cited_text(value: object) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(v) for v in value)
    return value or ""


def _extract_pass1_text(raw_response: str | None) -> str:
    """Return only pass1_text from the stored raw_response JSON blob."""
    if not raw_response:
        return ""
    try:
        data = json.loads(raw_response)
        return data.get("pass1_text") or ""
    except Exception:
        return raw_response


def _extract_value(stored: dict, question: "Question") -> object:
    """Reconstruct the typed answer value from the stored payload fields."""
    qt = question.question_type
    if qt == "boolean":
        return stored.get("value_boolean")
    if qt == "categorical":
        if question.allow_multiple:
            return stored.get("value_categorical_multi") or []
        return stored.get("value_categorical")
    return stored.get("value_text")


class SeerClient:
    def __init__(
        self,
        api_base: str,
        api_token: str,
        review_id: int | None = None,
        questions: "list[Question] | None" = None,
    ) -> None:
        # api_base is expected to end in /api/v1 (no trailing slash)
        self._base = api_base.rstrip("/")
        # Map version_id → Question for value extraction and key lookup
        self._question_map: dict[int, "Question"] = (
            {q.version_id: q for q in questions} if questions else {}
        )
        self._headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._headers, timeout=60)

    async def fetch_ocr_markdown(self, paper_id: int) -> str | None:
        """Return OCR markdown for paper_id, or None if unavailable."""
        url = f"{self._base}/papers/{paper_id}/ocr/"
        async with self._client() as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.get(url)
                except httpx.HTTPError as exc:
                    logger.warning("OCR fetch error paper=%d: %s", paper_id, exc)
                    if attempt == _MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(_BACKOFF_BASE ** attempt)
                    continue

                if resp.status_code == 404:
                    return None
                if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE ** attempt)
                    continue
                resp.raise_for_status()

                try:
                    data = resp.json()
                    return data.get("markdown") or None
                except Exception as exc:
                    logger.warning("OCR parse error paper=%d: %s", paper_id, exc)
                    return None

        return None

    async def post_answers_bulk(self, answers: list[dict]) -> None:
        """POST answers to /experiment-runs/{run_id}/answers/bulk/.

        Endpoint is idempotent: re-posting upserts on (run, paper, question_version).
        All answers in a single call must belong to the same run_id.
        """
        if not answers:
            return

        run_id = answers[0]["run"]
        url = f"{self._base}/experiment-runs/{run_id}/answers/bulk/"

        payload = []
        for stored in answers:
            version_id = stored.get("question_version")
            question = self._question_map.get(version_id)
            if question is None:
                logger.warning("Unknown question version_id=%s, skipping", version_id)
                continue

            item: dict = {
                "paper": stored["paper"],
                "question_key": question.key,
                "value": _extract_value(stored, question),
                "extraction_status": stored.get("extraction_status") or "ok",
                "extraction_detail": stored.get("extraction_detail") or "",
                "confidence": stored.get("confidence"),
                "comment": stored.get("comment") or "",
                "cited_text": _coerce_cited_text(stored.get("cited_text")),
                "cited_text_verified": stored.get("cited_text_verified"),
                "raw_response": _extract_pass1_text(stored.get("raw_response")),
                "tokens_total": stored.get("tokens_total") or 0,
                "tokens_input": stored.get("tokens_input") or 0,
                "tokens_output": stored.get("tokens_output") or 0,
                "tokens_cached": stored.get("tokens_cached") or 0,
                "latency_ms": stored.get("latency_ms") or 0,
            }
            if stored.get("cost"):
                # Django DecimalField: max_digits=12, decimal_places=6
                item["cost"] = f"{float(stored['cost']):.6f}"
                item["cost_currency"] = stored.get("cost_currency") or "USD"
            payload.append(item)

        if not payload:
            return

        async with self._client() as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.post(url, json=payload)
                except httpx.HTTPError as exc:
                    logger.warning("Bulk post error run=%s (attempt %d): %s", run_id, attempt, exc)
                    if attempt == _MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(_BACKOFF_BASE ** attempt)
                    continue

                if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE ** attempt)
                    continue

                resp.raise_for_status()
                result = resp.json()
                logger.info(
                    "Bulk posted run=%s: created=%s updated=%s errors=%s",
                    run_id,
                    result.get("created", "?"),
                    result.get("updated", "?"),
                    result.get("errors", []),
                )
                return


class DryRunSeerClient(SeerClient):
    """Prints payloads instead of posting; still fetches real OCR if reachable."""

    async def post_answers_bulk(self, answers: list[dict]) -> None:
        run_id = answers[0]["run"] if answers else "?"
        print(f"[dry-run] Would POST {len(answers)} answers to experiment-runs/{run_id}/answers/bulk/:")
        for stored in answers:
            version_id = stored.get("question_version")
            question = self._question_map.get(version_id)
            key = question.key if question else f"version_id={version_id}"
            value = _extract_value(stored, question) if question else "?"
            print(f"  paper={stored['paper']} q={key} value={value!r}")
