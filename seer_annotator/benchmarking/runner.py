"""Resumable, SQLite-only Phase-2 formatting benchmark runner.

This module intentionally knows nothing about SEER's source database or Django.
Its input is a frozen :class:`BenchmarkStore` dataset and its output is more
rows in that same local SQLite database.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable, Mapping

from ..annotate.parse import ANNOTATE_RESPONSE_FORMAT, parse_structured_output_diagnostic
from ..annotate.prompt import build_format_messages
from ..config import DEFAULT_REQUEST_TIMEOUT, Question, ProviderSettings, Settings
from ..llm import LLMResult, complete as llm_complete
from ..rate_limiter import PerProviderRateLimiter
from .store import BenchmarkStore, ModelConfig, is_secret_key


CompletionFn = Callable[..., Any]


def _question(raw: Mapping[str, Any]) -> Question:
    """Validate a frozen question, supplying harmless legacy defaults."""
    data = dict(raw)
    data.setdefault("question_id", 0)
    data.setdefault("version", 0)
    data.setdefault("version_id", 0)
    data.setdefault("label", data.get("key", ""))
    data.setdefault("question_type", "text")
    return Question.model_validate(data)


def _without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _without_secrets(v)
            for k, v in value.items()
            if not is_secret_key(k)
        }
    if isinstance(value, (list, tuple)):
        return [_without_secrets(v) for v in value]
    return value


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _result_parts(result: Any) -> tuple[str, dict[str, Any], float | None, int | None, dict[str, Any]]:
    if isinstance(result, str):
        return result, {}, None, None, {}
    value = _plain(result)
    if isinstance(value, Mapping):
        text = value.get("text")
        if text is None:
            content = value.get("content")
            text = content if isinstance(content, str) else json.dumps(content or value)
        usage = _plain(value.get("usage") or {})
        cost = value.get("cost")
        latency = value.get("latency_ms")
        raw = _without_secrets(value.get("raw") or {})
        return str(text or ""), dict(usage) if isinstance(usage, Mapping) else {}, _number(cost), latency, raw
    text = str(getattr(result, "text", "") or "")
    usage = _plain(getattr(result, "usage", None) or {})
    cost = _number(getattr(result, "cost", None))
    latency = getattr(result, "latency_ms", None)
    raw = _without_secrets(getattr(result, "raw", {}) or {})
    return text, dict(usage) if isinstance(usage, Mapping) else {}, cost, latency, raw


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value if not isinstance(value, Decimal) else value)
    except (TypeError, ValueError):
        return None


class BenchmarkRunner:
    """Run one stored model configuration over a frozen benchmark dataset."""

    def __init__(
        self,
        store: BenchmarkStore,
        *,
        settings: Settings | None = None,
        settings_path: str | None = None,
        completion_fn: CompletionFn | None = None,
        concurrency: int = 1,
        per_provider_rpm: float | None = None,
        stale_after_seconds: float = 3600.0,
    ) -> None:
        self.store = store
        self.settings = settings or (Settings.load(settings_path) if settings_path else Settings())
        self._uses_default_completion = completion_fn is None
        self.completion_fn = completion_fn if completion_fn is not None else llm_complete
        self.concurrency = max(1, int(concurrency))
        self.rate_limiter = PerProviderRateLimiter(per_provider_rpm)
        self.stale_after_seconds = stale_after_seconds
        self.run_token = uuid.uuid4().hex

    def _provider_params(self, config: ModelConfig) -> dict[str, Any]:
        provider = self.settings.providers.get(config.provider, ProviderSettings())
        params = dict(_without_secrets(config.params) or {})
        api_key = provider.resolved_api_key()
        if api_key:
            params["api_key"] = api_key
        # Provider settings supply the endpoint defaults, but an explicit value in
        # the model config wins: one Azure provider entry can then serve
        # deployments needing different api_versions (e.g. gpt-5-* require a
        # newer version than gpt-4-era deployments).
        if provider.base_url:
            params.setdefault("api_base", provider.base_url)
        if provider.api_version:
            params.setdefault("api_version", provider.api_version)
        params.setdefault("timeout", config.timeout if config.timeout is not None else DEFAULT_REQUEST_TIMEOUT)
        if config.temperature is not None:
            params.setdefault("temperature", config.temperature)
        return params

    async def _call(self, config: ModelConfig, messages: list[dict]) -> Any:
        await self.rate_limiter.acquire(config.provider)
        kwargs = self._provider_params(config)
        response_format = ANNOTATE_RESPONSE_FORMAT if config.structured_output else None
        # ``drop_params`` is a LiteLLM process setting, not a provider API
        # parameter. Keep injected completion functions deterministic.
        if self._uses_default_completion:
            import litellm
            litellm.drop_params = config.drop_params
        result = self.completion_fn(
            config.model, config.provider, messages,
            response_format=response_format, **kwargs,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_one(self, execution: Mapping[str, Any], config: ModelConfig, semaphore: asyncio.Semaphore,
                       *, retry_errors: bool) -> str:
        execution_id = int(execution["id"])
        if not self.store.claim_execution(execution_id, retry_errors=retry_errors, run_token=self.run_token):
            return "skipped"
        case_rows = self.store.get_cases_by_id(int(execution["case_id"]))
        if case_rows is None:
            try:
                self.store.save_execution(execution_id, status="error", error="case no longer exists",
                                          run_token=self.run_token)
            except RuntimeError:
                return "skipped"
            return "error"
        try:
            question_data = json.loads(case_rows["questions_json"])
            questions = [_question(q) for q in question_data]
            messages = build_format_messages(case_rows["pass1_text"], questions, require_status=True)
            async with semaphore:
                started = time.monotonic()
                response = await self._call(config, messages)
                elapsed_ms = int((time.monotonic() - started) * 1000)
            text, usage, cost, result_latency, raw = _result_parts(response)
            diagnostics = parse_structured_output_diagnostic(text, [q.key for q in questions], annotate_mode=True)
            diagnostics["raw_metadata"] = raw
            self.store.save_execution(
                execution_id, status="complete", raw_response=text,
                parsed_answers=diagnostics["answers"], diagnostics=diagnostics, usage=usage,
                cost=cost, latency_ms=result_latency if result_latency is not None else elapsed_ms,
                run_token=self.run_token,
            )
            return "complete"
        except Exception as exc:
            try:
                self.store.save_execution(execution_id, status="error", error=f"{type(exc).__name__}: {exc}",
                                          run_token=self.run_token)
            except RuntimeError:
                return "skipped"
            return "error"

    async def run(self, dataset: str, config: ModelConfig, *, retry_errors: bool = False) -> dict[str, int]:
        config_id = self.store.add_model_config(config)
        self.store.recover_stale_running(self.stale_after_seconds)
        self.store.ensure_executions(dataset, config_id)
        rows = self.store.get_executions(dataset, config_id)
        work = [
            row for row in rows
            if row["status"] == "pending" or (retry_errors and row["status"] == "error")
        ]
        semaphore = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(
            self._run_one(row, config, semaphore, retry_errors=retry_errors) for row in work
        ))
        summary = {"complete": 0, "error": 0, "skipped": len(rows) - len(work)}
        for status in results:
            summary[status] = summary.get(status, 0) + 1
        return summary

    def run_sync(self, dataset: str, config: ModelConfig, *, retry_errors: bool = False) -> dict[str, int]:
        return asyncio.run(self.run(dataset, config, retry_errors=retry_errors))


async def run_benchmark(
    store: BenchmarkStore, dataset: str, config: ModelConfig, **kwargs: Any,
) -> dict[str, int]:
    """Functional convenience wrapper used by CLIs and small scripts."""
    retry_errors = bool(kwargs.pop("retry_errors", False))
    return await BenchmarkRunner(store, **kwargs).run(dataset, config, retry_errors=retry_errors)
