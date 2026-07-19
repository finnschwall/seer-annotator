"""Async batch API execution for Pass-1 and/or Pass-2 requests.

Supports Anthropic (beta.messages.batches) and OpenAI (files + batches).
Google Gemini uses BigQuery/GCS and is architecturally different — not supported.

Each pass is independently toggleable via batch_p1 / batch_p2 in settings.toml [run_defaults] or per-run RunConfig.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import logging
from decimal import Decimal
from typing import Callable, Protocol, runtime_checkable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)
_console = Console()


def _format_llm_error(exc: BaseException) -> str:
    """Render an LLM-call exception with everything LiteLLM/httpx tucked away on it.

    LiteLLM's mapped exceptions (APIError, RateLimitError, ...) often carry an
    empty ``str(exc)`` — the useful detail lives in ``litellm_debug_info``, the
    wrapped HTTP response body, or ``__cause__`` instead. Without pulling those
    out, the log line just says e.g. "APIError: litellm.APIError: AzureException
    APIError - " with nothing after the dash.
    """
    parts = [f"{type(exc).__name__}: {exc}"]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")

    debug_info = getattr(exc, "litellm_debug_info", None)
    if debug_info:
        parts.append(f"debug_info={debug_info}")

    response = getattr(exc, "response", None)
    body = getattr(exc, "body", None)
    for label, obj in (("response", response), ("body", body)):
        if obj is None:
            continue
        text = getattr(obj, "text", None)
        parts.append(f"{label}={text if text is not None else obj}")

    cause = exc.__cause__
    if cause is not None and str(cause):
        parts.append(f"caused by {type(cause).__name__}: {cause}")

    return " | ".join(parts)

_DEFAULT_MAX_TOKENS_P1 = 4096
_DEFAULT_MAX_TOKENS_P2 = 1024


class BatchPendingError(Exception):
    """Raised when a submitted batch has been polled once and is not yet terminal.

    Batch mode is observable-by-design: submission and polling are decoupled so a
    single poll check never blocks a worker for the up-to-24h a provider batch SLA
    allows. Whoever drives the pipeline (e.g. ``experiments/llm_runner.py`` on the
    SEER side) is expected to catch this, persist/update an observable record keyed
    on ``batch_id``, and re-invoke the pipeline later (e.g. from a
    ``poll_llm_batches`` scheduled command) — thanks to the kv-based resumability in
    ``submit_and_poll`` (``store.get_batch_id``/``save_batch_id``), re-entering the
    pipeline from scratch re-submits nothing and simply polls once more.
    """

    def __init__(
        self,
        batch_id: str,
        label: str,
        status: str,
        *,
        pass_name: str | None = None,
        request_count: int | None = None,
    ) -> None:
        self.batch_id = batch_id
        self.label = label
        self.status = status
        self.pass_name = pass_name
        self.request_count = request_count
        super().__init__(f"Batch {batch_id} ({label}) not yet done: status={status}")


# ---------------------------------------------------------------------------
# Batch cost calculation
# ---------------------------------------------------------------------------
#
# Batch collectors only ever get token counts back from the provider — the
# Anthropic/OpenAI batch APIs don't return a dollar cost the way a sync
# completion response's usage block can be fed straight into
# litellm.completion_cost() (see llm.py). This mirrors that sync-path
# calculation but works from a plain {model, usage} dict instead of a real
# response object, and forces litellm's batch-discounted pricing branch via
# call_type="retrieve_batch" (batch APIs are typically ~50% off standard
# per-token pricing).

def _batch_item_cost(provider: str, model: str, usage: dict) -> Decimal | None:
    """Compute one collected batch item's cost from its token counts, or None."""
    import litellm

    litellm_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_write_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_tokens", 0),
    }
    try:
        cost_float = litellm.completion_cost(
            completion_response={"model": f"{provider}/{model}", "usage": litellm_usage},
            call_type="retrieve_batch",
        )
    except Exception:
        return None
    if cost_float is None:
        return None
    return Decimal(str(cost_float))


def _add_batch_costs(usage_by_cid: dict[str, dict], provider: str, model: str) -> None:
    """Add a "cost" key to each usage dict in place, best-effort."""
    for usage in usage_by_cid.values():
        cost = _batch_item_cost(provider, model, usage)
        if cost is not None:
            usage["cost"] = cost


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BatchProvider(Protocol):
    def submit(self, requests: list[dict]) -> str: ...
    def poll(self, batch_id: str) -> str: ...  # "running" | "done" | "failed"
    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
        """Return ({cid: text}, {cid: usage_stats}, {cid: error_detail}).

        ``error_detail`` is populated only for custom_ids that did NOT succeed
        (canceled/errored/expired items) — such cids are intentionally absent from
        the first two dicts so callers can't mistake a failure for an empty-string
        success.
        """
        ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

class AnthropicBatchProvider:
    def __init__(self, api_key: str | None = None) -> None:
        import anthropic
        self._client = anthropic.Anthropic(**({"api_key": api_key} if api_key else {}))

    def submit(self, requests: list[dict]) -> str:
        try:
            batch = self._client.beta.messages.batches.create(requests=requests)
        except Exception as exc:
            body = getattr(exc, "body", None)
            if body:
                logger.error("Anthropic batch submit failed: %s", body)
            raise
        return batch.id

    def prewarm(self, params: dict) -> None:
        """Send a ``max_tokens=0`` cache pre-warm request as a plain synchronous call.

        This deliberately bypasses submit()/poll()/collect() — i.e. the Message
        Batches API — entirely. Anthropic's documented pre-warming pattern
        (https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pre-warming-the-cache)
        requires ``max_tokens: 0`` so the API reads the prefix, writes the cache,
        and returns immediately without generating output. But the same docs list
        Message Batches as one of the contexts where ``max_tokens: 0`` is
        *rejected* (alongside streaming, extended thinking, structured outputs,
        and forced tool_choice) — a batch item still needs ``max_tokens >= 1``
        like any other request.

        Concretely: submitting a pre-warm item through ``submit()`` gets the
        batch accepted, but that one item comes back from ``collect()`` as
        ``result.type == "errored"`` with ``invalid_request_error: max_tokens:
        must be greater than or equal to 1`` — silently defeating the pre-warm
        (its result is discarded either way, so nothing crashes, but the cache
        is never actually warmed). Only a direct, non-batch ``messages.create()``
        call — what this method does — accepts ``max_tokens: 0``.

        Best-effort: failures are logged and swallowed rather than raised, since
        a failed pre-warm just means the main batch proceeds without a warm
        cache (see the caller in ``_execute_pass1_with_groups``), not a reason
        to abort the run.
        """
        try:
            self._client.messages.create(**params)
        except Exception as exc:
            logger.warning(
                "Anthropic cache pre-warm request failed (continuing without a "
                "warm cache): %s",
                _format_llm_error(exc),
            )

    def poll(self, batch_id: str) -> str:
        batch = self._client.beta.messages.batches.retrieve(batch_id)
        # processing_status is one of "in_progress" | "canceling" | "ended" (see
        # anthropic.types.beta.messages.beta_message_batch.BetaMessageBatch). Only
        # "ended" is terminal — "canceling" means a cancellation was requested but
        # in-flight requests may still complete, so the batch is NOT done yet.
        # There is no batch-level "failed" status at all: once "ended", per-item
        # outcomes (succeeded/errored/canceled/expired) are read in collect(), not
        # here — request_counts.canceled/.errored > 0 on an "ended" batch is normal,
        # not a reason to treat the batch itself as failed.
        if batch.processing_status == "ended":
            return "done"
        return "running"

    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
        results: dict[str, str] = {}
        usage_by_cid: dict[str, dict] = {}
        errors_by_cid: dict[str, str] = {}
        total_input = 0
        total_output = 0
        cache_create = 0
        cache_read = 0
        n_succeeded = 0
        n_failed = 0

        for item in self._client.beta.messages.batches.results(batch_id):
            cid = item.custom_id
            if item.result.type == "succeeded":
                msg = item.result.message
                text = "".join(
                    block.text for block in msg.content if hasattr(block, "text")
                )
                results[cid] = text
                n_succeeded += 1
                if hasattr(msg, "usage") and msg.usage:
                    u = msg.usage
                    inp  = getattr(u, "input_tokens", 0) or 0
                    out  = getattr(u, "output_tokens", 0) or 0
                    cw   = getattr(u, "cache_creation_input_tokens", 0) or 0
                    cr   = getattr(u, "cache_read_input_tokens", 0) or 0
                    total_input  += inp
                    total_output += out
                    cache_create += cw
                    cache_read   += cr
                    usage_by_cid[cid] = {
                        "input_tokens": inp,
                        "output_tokens": out,
                        "cache_write_tokens": cw,
                        "cache_read_tokens": cr,
                    }
            else:
                # Non-succeeded item.result.type is one of "errored" | "canceled" |
                # "expired" (see beta_message_batch_result.py). Do NOT write an
                # empty-string sentinel into `results` — that would look like a
                # succeeded-but-empty response to callers. Surface the real detail
                # instead so it can be posted as a proper error answer/resolution.
                if item.result.type == "errored":
                    err_resp = getattr(item.result, "error", None)
                    beta_error = getattr(err_resp, "error", None)
                    err_type = getattr(beta_error, "type", None)
                    err_msg = getattr(beta_error, "message", None)
                    detail = f"{err_type}: {err_msg}" if err_msg else (str(err_resp) if err_resp else "batch item errored")
                else:
                    detail = f"batch item {item.result.type}"
                errors_by_cid[cid] = detail
                logger.warning("Batch item %s: result type=%s — %s", cid, item.result.type, detail)
                n_failed += 1

        cache_total = cache_create + cache_read
        logger.info(
            "Batch %s usage — succeeded=%d failed=%d | "
            "input=%d output=%d | cache_write=%d cache_read=%d "
            "(%.1f%% cache hit rate; note: batch parallelism causes multiple writes — expected)",
            batch_id, n_succeeded, n_failed,
            total_input, total_output,
            cache_create, cache_read,
            100.0 * cache_read / max(cache_total, 1),
        )
        return results, usage_by_cid, errors_by_cid


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIBatchProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def submit(self, requests: list[dict]) -> str:
        try:
            lines = "\n".join(json.dumps(r) for r in requests)
            file_obj = self._client.files.create(
                file=("batch.jsonl", io.BytesIO(lines.encode()), "application/jsonl"),
                purpose="batch",
            )
            batch = self._client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            return batch.id
        except Exception as exc:
            body = getattr(exc, "body", None)
            if body:
                logger.error("OpenAI/Azure batch submit failed: %s", body)
            raise

    def poll(self, batch_id: str) -> str:
        batch = self._client.batches.retrieve(batch_id)
        # status is one of "validating" | "failed" | "in_progress" | "finalizing" |
        # "completed" | "expired" | "cancelling" | "cancelled" (see
        # openai.types.batch.Batch — note both "cancelling"/"cancelled" are spelled
        # with a double "l"). Only "completed" means results are ready: "finalizing"
        # is still writing the output file (output_file_id may not be populated
        # yet — treating it as done risked collect() reading an empty/missing
        # file), and "cancelling" is in-flight cancellation, not yet terminal.
        if batch.status == "completed":
            return "done"
        if batch.status in ("failed", "expired", "cancelled"):
            return "failed"
        return "running"

    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
        batch = self._client.batches.retrieve(batch_id)
        results: dict[str, str] = {}
        usage_by_cid: dict[str, dict] = {}
        errors_by_cid: dict[str, str] = {}

        output_file_id = batch.output_file_id
        if output_file_id:
            raw = self._client.files.content(output_file_id).read().decode()
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj["custom_id"]
                    text = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
                    results[cid] = text
                    u = obj.get("response", {}).get("body", {}).get("usage", {})
                    if u:
                        usage_by_cid[cid] = {
                            "input_tokens": u.get("prompt_tokens", 0) or 0,
                            "output_tokens": u.get("completion_tokens", 0) or 0,
                            "cache_write_tokens": 0,
                            "cache_read_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                        }
                except (KeyError, IndexError, json.JSONDecodeError) as exc:
                    logger.warning("Could not parse batch output line: %s", exc)
        else:
            logger.info("Batch %s has no output_file_id (no successful requests?)", batch_id)

        # Per-request failures land in a SEPARATE file (error_file_id), same JSONL
        # shape as the output file but with an "error" object instead of "response"
        # (roughly {"custom_id", "response": null, "error": {"code", "message"}}).
        # Previously unread entirely — a request that failed and landed only here
        # never appeared in `results` OR anywhere else the caller iterates, so it
        # sat "pending forever" from the caller's point of view.
        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            raw_err = self._client.files.content(error_file_id).read().decode()
            for line in raw_err.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj["custom_id"]
                except (KeyError, json.JSONDecodeError) as exc:
                    logger.warning("Could not parse batch error-file line: %s", exc)
                    continue
                err = obj.get("error") or {}
                resp = obj.get("response") or {}
                resp_body_error = ((resp.get("body") or {}).get("error") or {}) if resp else {}
                message = err.get("message") or resp_body_error.get("message")
                code = err.get("code") or resp_body_error.get("code")
                detail = f"{code}: {message}" if message else f"batch item error (custom_id={cid})"
                errors_by_cid[cid] = detail
                logger.warning("Batch item %s errored: %s", cid, detail)

        if not output_file_id and not error_file_id:
            logger.error("Batch %s has neither output_file_id nor error_file_id", batch_id)

        return results, usage_by_cid, errors_by_cid


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def make_provider(provider_name: str, api_key: str | None, base_url: str | None) -> BatchProvider:
    p = provider_name.lower()
    if p == "anthropic":
        return AnthropicBatchProvider(api_key=api_key)
    if p in ("openai", "azure"):
        return OpenAIBatchProvider(api_key=api_key, base_url=base_url)
    raise ValueError(
        f"Batch mode is not supported for provider {provider_name!r}. "
        "Supported providers: anthropic, openai, azure."
    )


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def _extract_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull the first system message out of the messages list (for Anthropic)."""
    system: str | None = None
    rest: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system" and system is None:
            content = msg.get("content", "")
            system = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        else:
            rest.append(msg)
    return system, rest


def build_p1_request(
    *,
    custom_id: str,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float | None,
    model_params: dict,
) -> dict:
    p = provider.lower()
    max_tokens = model_params.get("max_tokens", _DEFAULT_MAX_TOKENS_P1)
    extra = {k: v for k, v in model_params.items() if k != "max_tokens"}

    if p == "anthropic":
        system, user_messages = _extract_system(messages)
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        params.update(extra)
        if system:
            params["system"] = system
        return {"custom_id": custom_id, "params": params}

    # openai / azure
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra)
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def build_p2_request(
    *,
    custom_id: str,
    provider: str,
    model: str,
    messages: list[dict],
    model_params: dict,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> dict:
    p = provider.lower()
    max_tokens = model_params.get("max_tokens", _DEFAULT_MAX_TOKENS_P2)
    extra = {k: v for k, v in model_params.items() if k != "max_tokens"}

    if p == "anthropic":
        # Anthropic batch uses tool_choice for structured output; response_format is not supported.
        system, user_messages = _extract_system(messages)
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if temperature is not None:
            params["temperature"] = temperature
        params.update(extra)
        if system:
            params["system"] = system
        return {"custom_id": custom_id, "params": params}

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra)
    if response_format is not None:
        body["response_format"] = response_format
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


# ---------------------------------------------------------------------------
# Single-shot poll (no loop — see BatchPendingError)
# ---------------------------------------------------------------------------

def _poll_once(
    provider: BatchProvider,
    batch_id: str,
    batch_key: str,
    store,
    label: str = "Batch",
    *,
    pass_name: str | None = None,
    request_count: int | None = None,
) -> None:
    """Check batch status exactly once — no sleep loop.

    A submitted batch may take up to a provider's SLA (typically <=24h) to reach a
    terminal state; blocking synchronously inside the worker's event loop for that
    long holds the worker hostage. Instead: check once, and either return (batch
    done, caller proceeds to collect()), raise RuntimeError (batch failed — kv
    cleared so it won't be endlessly re-polled), or raise BatchPendingError (still
    running — kv left in place so the next external call resumes polling the same
    batch_id, e.g. via a `poll_llm_batches` scheduled command).
    """
    result = provider.poll(batch_id)
    logger.info("Batch %s: %s", batch_id, result)
    if result == "done":
        return
    if result == "failed":
        # Only cleared on success before this fix — a failed batch's kv entry was
        # never removed, so it would be re-polled forever on retry.
        store.delete_batch_id(batch_key)
        _console.print(f"[bold red]{label}[/] {batch_id} failed/expired/canceled")
        raise RuntimeError(f"Batch {batch_id} failed/expired/canceled")
    _console.print(f"[bold]{label}[/] {batch_id} [dim]still {result} — will resume later[/]")
    raise BatchPendingError(batch_id, label, result, pass_name=pass_name, request_count=request_count)


async def submit_and_poll(
    provider: BatchProvider,
    requests: list[dict],
    store,
    batch_key: str,
    label: str = "Batch",
    *,
    pass_name: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> tuple[dict[str, str], dict[str, dict], dict[str, str]]:
    """Submit a batch (or resume an existing one), then poll it exactly once.

    Returns ({custom_id: text}, {custom_id: usage_stats}, {custom_id: error_detail}).
    Raises ``BatchPendingError`` if the batch is submitted but not yet terminal —
    callers must let this propagate (it is the signal to park the job and resume
    later) rather than catching it here.

    When ``provider_name``/``model`` are given, each usage dict is annotated
    with a best-effort "cost" key (see ``_add_batch_costs``); omit them (as the
    cache pre-warm call does — its result is discarded anyway) to skip this.
    """
    batch_id = store.get_batch_id(batch_key)
    if batch_id:
        _console.print(f"[bold]{label}[/] resuming [dim]{batch_id}[/]")
        logger.info("Resuming existing batch %s (key=%s)", batch_id, batch_key)
    else:
        batch_id = provider.submit(requests)
        store.save_batch_id(batch_key, batch_id)
        _console.print(f"[bold]{label}[/] submitted {len(requests)} request(s) → [dim]{batch_id}[/]")
        logger.info("Submitted batch %s (key=%s, n=%d)", batch_id, batch_key, len(requests))

    _poll_once(
        provider, batch_id, batch_key, store, label=label,
        pass_name=pass_name, request_count=len(requests),
    )

    results, usage_by_cid, errors_by_cid = provider.collect(batch_id)
    store.delete_batch_id(batch_key)
    if provider_name and model:
        _add_batch_costs(usage_by_cid, provider_name, model)
    n_ok = len(results)
    n_err = len(errors_by_cid)
    _console.print(f"[bold]{label}[/] [green]✓[/] complete — {n_ok} result(s), {n_err} error(s)")
    logger.info("Batch %s complete: %d results, %d errors", batch_id, n_ok, n_err)
    return results, usage_by_cid, errors_by_cid


# ---------------------------------------------------------------------------
# Main batch pipeline
# ---------------------------------------------------------------------------

def _p1_dump_path(dump_dir: str, run_id: int) -> "pathlib.Path":
    import pathlib
    return pathlib.Path(dump_dir) / f"p1_{run_id}.json"


def _save_p1_dump(dump_dir: str, run_id: int, p1_texts: dict) -> None:
    import pathlib, json as _json
    p = _p1_dump_path(dump_dir, run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(p1_texts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("P1 results saved to %s (delete after verifying run is complete)", p)


def _load_p1_dump(dump_dir: str, run_id: int) -> "dict | None":
    import pathlib, json as _json
    p = _p1_dump_path(dump_dir, run_id)
    if p.exists():
        data = _json.loads(p.read_text(encoding="utf-8"))
        logger.info("Loaded P1 dump from %s (%d entries) — skipping batch re-submission", p, len(data))
        return data
    return None


async def _execute_pass1_with_groups(
    run,
    cfg,
    papers,
    source_texts: dict,
    pending_cells: dict,
    store,
    settings,
    dry_run: bool,
    groups_def,
    *,
    sem: "asyncio.Semaphore | None" = None,
    limiter=None,
    on_cell_done: "Callable[[], None] | None" = None,
    on_total_known: "Callable[[int], None] | None" = None,
    build_p1_messages: "Callable[[str, list, int], list[dict]] | None" = None,
    should_skip_cell: "Callable[[int, int, int], bool] | None" = None,
) -> "tuple[dict, dict, dict, str | None]":
    """Internal implementation of _execute_pass1 that also receives groups_def.

    Returns ``(p1_texts, p1_usage, p1_errors_by_cid, fatal_error)`` — ``p1_errors_by_cid``
    is ``{cid: error_detail}`` for cells that failed with an attributable per-item
    error (batch item errored/canceled/expired, or an online call that raised);
    such cids are absent from ``p1_texts``. ``fatal_error`` (unchanged from before)
    signals a whole-run-aborting failure such as a bad model/API key, not a
    per-cell issue — kept as a separate, rarer signal from ``p1_errors_by_cid``.

    ``groups_def`` is either a single ``list[list[Question]]`` applied to every
    paper (annotation's fixed question grouping), or a ``dict[paper_id, list[list[Question]]]``
    for callers whose groups vary per paper (e.g. arbitration, where each paper
    only has its own disputed questions).

    ``build_p1_messages``, when given, replaces the default ``build_messages(...)``
    call for constructing the Pass-1 messages — ``(source_text, group, paper_id) -> messages``. ``paper_id`` lets callers whose
    extra context varies per paper (e.g. arbitration's candidates) look it up without
    needing a separate closure per paper.

    ``should_skip_cell``, when given, replaces ``store.should_skip_cell`` for the
    dedup/resume check — ``(run_id, paper_id, version_id) -> bool``. Annotation's
    default checks the ``answers`` table; arbitration passes a checker over the
    ``resolutions`` table instead.
    """
    from .config import DEFAULT_REQUEST_TIMEOUT, ProviderSettings
    from .caching import apply_cache
    from .annotate.prompt import build_messages as _default_build_messages
    from .llm import complete as llm_complete, dummy_complete

    _base_complete = dummy_complete if dry_run else llm_complete

    prov_settings = settings.providers.get(run.model_provider, ProviderSettings())
    p1_api_key = prov_settings.resolved_api_key()
    p1_base_url = prov_settings.base_url

    effective_system = cfg.system_prompt
    batch_p1 = cfg.batch_p1
    _skip_cell = should_skip_cell if should_skip_cell is not None else store.should_skip_cell

    def _build_messages(source: str, group: list, paper_id: int) -> list[dict]:
        if build_p1_messages is not None:
            return build_p1_messages(source, group, paper_id)
        return _default_build_messages(
            source,
            group,
            text_source=cfg.text_source,
            system_prompt=effective_system,
            cache_first=cfg.cache_first,
        )

    # Build pending_cells in-place (unified path — replaces the duplicate blocks
    # that previously existed in the batch_p1 and online branches).
    pending_cells.clear()
    for paper in papers:
        if paper.paper_id not in source_texts:
            continue
        paper_groups = groups_def.get(paper.paper_id, []) if isinstance(groups_def, dict) else groups_def
        for group_idx, group in enumerate(paper_groups):
            cells = [(run.run_id, paper.paper_id, q.version_id) for q in group]
            if all(_skip_cell(*c) for c in cells):
                continue
            cid = f"{run.run_id}-{paper.paper_id}-{group_idx}"
            pending_cells[cid] = (paper, group, group_idx)

    if on_total_known is not None:
        on_total_known(len(pending_cells))

    if batch_p1 and not dry_run:
        # Try loading a saved P1 dump (crash-safe resume without re-billing)
        p1_texts: dict[str, str] = _load_p1_dump(settings.runtime.p1_dump_dir, run.run_id) or {}
        p1_usage: dict[str, dict] = {}  # {cid: usage_stats} — empty when loaded from dump
        p1_errors: dict[str, str] = {}  # {cid: error_detail} — empty when loaded from dump (no way to know)

        if not p1_texts:
            if not pending_cells:
                logger.info("Run %d: no pending P1 work", run.run_id)
            else:
                p1_requests = []
                first_messages: list[dict] | None = None
                for cid, (paper, group, _) in pending_cells.items():
                    source = source_texts[paper.paper_id]
                    messages = _build_messages(source, group, paper.paper_id)
                    messages = apply_cache(run.model_provider, messages, cfg.cache, ttl=cfg.cache_ttl)
                    if first_messages is None:
                        first_messages = messages
                    p1_requests.append(build_p1_request(
                        custom_id=cid,
                        provider=run.model_provider,
                        model=run.model_name,
                        messages=messages,
                        temperature=cfg.temperature,
                        model_params=cfg.model_params,
                    ))

                provider = make_provider(run.model_provider, p1_api_key, p1_base_url)

                # Cache pre-warm: write the shared prefix to the 1h cache before the
                # main batch starts, so every real request hits a warm cache instead
                # of racing to write it in parallel (which caused ~48 writes vs 1 in
                # testing — batch items aren't processed sequentially, so without
                # this every item in the main batch could race to write the same
                # cache entry at once). Only useful when there IS a prefix genuinely
                # shared across every paper's request: `cache_first="questions"`
                # places the (paper-independent) questions/instructions block first
                # — the block `caching.py` actually marks with `cache_control` — so
                # that's what a pre-warm can usefully prime. `cache_first="text"`
                # marks the paper's own (paper-*specific*) text instead, which a
                # single once-per-run pre-warm can't help across different papers —
                # see ISSUE.md for that remaining gap (relevant with per-question/
                # grouped `batching`, where it could still help *within* one paper).
                #
                # Anthropic-only: this is Anthropic's documented explicit
                # cache_control pre-warm pattern (see AnthropicBatchProvider.prewarm
                # for the full explanation and why it must NOT go through
                # submit_and_poll/Message Batches). `apply_cache` is a no-op for
                # openai/azure (their prefix caching is automatic/server-side), so
                # building a pre-warm item for those providers would just be a
                # wasted "warmup" call with nothing to prime.
                main_key = f"{run.run_id}:p1"
                use_prewarm = (
                    cfg.cache
                    and cfg.cache_first == "questions"
                    and len(p1_requests) > 1
                    and run.model_provider.lower() == "anthropic"
                )
                if use_prewarm and store.get_batch_id(main_key):
                    # We're resuming (e.g. via a scheduled poll re-entering this whole
                    # function after a prior BatchPendingError) and the MAIN batch is
                    # already submitted/in-flight from that earlier invocation. A cache
                    # write now couldn't retroactively help requests already queued at
                    # submission time anyway, so there's nothing useful left to warm —
                    # skip straight to resuming the main batch below; `store.get_batch_id`
                    # already has its id, so `submit_and_poll` will just poll/collect it.
                    use_prewarm = False
                if use_prewarm:
                    # A synthetic pre-warm request — NOT a real paper's request — per
                    # Anthropic's documented cache pre-warming pattern: max_tokens=0,
                    # a placeholder final (uncached) message, and the same cache_control
                    # breakpoint a real request would use. The API reads the prefix,
                    # writes the cache, and returns immediately with empty content and
                    # no generated output — there is no real answer to lose here, unlike
                    # the previous design (which spent one real paper's own request on
                    # this and merged whatever came back into the results — fragile
                    # across a park/resume cycle, since that merge only happened if
                    # both the pre-warm and the main batch finished within the same
                    # call: a resumed run skips re-submitting the pre-warm, per above,
                    # so that real paper's only answer was silently lost. See the
                    # commit/changelog for the bug this replaced).
                    assert first_messages is not None  # pending_cells was non-empty
                    prewarm_messages = copy.deepcopy(first_messages)
                    prewarm_messages[-1]["content"] = "warmup"
                    prewarm_request = build_p1_request(
                        custom_id=f"{run.run_id}-prewarm",
                        provider=run.model_provider,
                        model=run.model_name,
                        messages=prewarm_messages,
                        temperature=cfg.temperature,
                        model_params={**cfg.model_params, "max_tokens": 0},
                    )
                    logger.info(
                        "Run %d: sending synchronous max_tokens=0 pre-warm request "
                        "(outside the batch API — see AnthropicBatchProvider.prewarm) "
                        "to warm 1h cache before main batch",
                        run.run_id,
                    )
                    # NOT submit_and_poll: Message Batches rejects max_tokens=0 outright
                    # (a batch item still needs max_tokens >= 1), so this must be a
                    # plain synchronous messages.create() call instead — see
                    # AnthropicBatchProvider.prewarm's docstring for the full story,
                    # including the "max_tokens: must be greater than or equal to 1"
                    # error this replaces. Nothing from the response is ever used —
                    # it's a cache side effect only — so prewarm() itself discards
                    # the result; nothing here is merged into p1_texts/p1_usage/p1_errors.
                    assert isinstance(provider, AnthropicBatchProvider)
                    provider.prewarm(prewarm_request["params"])

                p1_texts, p1_usage, p1_errors = await submit_and_poll(
                    provider, p1_requests, store, main_key,
                    label="Batch P1", pass_name="p1",
                    provider_name=run.model_provider, model=run.model_name,
                )

                _save_p1_dump(settings.runtime.p1_dump_dir, run.run_id, p1_texts)
    else:
        # P1 online: gather all papers concurrently
        p1_texts = {}
        p1_usage = {}
        p1_errors = {}
        p1_tasks = []
        p1_task_cids: list[str] = []

        for cid, (paper, group, group_idx) in pending_cells.items():
            source = source_texts[paper.paper_id]

            async def _run_p1(cid=cid, paper=paper, group=group, source=source):
                try:
                    messages = _build_messages(source, group, paper.paper_id)
                    messages = apply_cache(run.model_provider, messages, cfg.cache, ttl=cfg.cache_ttl)
                    p1_kwargs: dict = {
                        "timeout": cfg.request_timeout or DEFAULT_REQUEST_TIMEOUT,
                    }
                    if cfg.temperature is not None:
                        p1_kwargs["temperature"] = cfg.temperature
                    if cfg.reasoning_effort:
                        p1_kwargs["reasoning_effort"] = cfg.reasoning_effort
                    p1_kwargs.update(cfg.model_params)
                    if p1_api_key:
                        p1_kwargs["api_key"] = p1_api_key
                    if p1_base_url:
                        p1_kwargs["api_base"] = p1_base_url
                    if limiter is not None:
                        await limiter.acquire(run.model_provider)
                    if sem is not None:
                        async with sem:
                            result = await _base_complete(
                                run.model_name, run.model_provider, messages, **p1_kwargs
                            )
                    else:
                        result = await _base_complete(
                            run.model_name, run.model_provider, messages, **p1_kwargs
                        )
                    usage = {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "cache_read_tokens": result.usage.cached_tokens,
                        "cost": result.cost,
                        "latency_ms": result.latency_ms,
                    }
                    return cid, result.text, usage
                finally:
                    if on_cell_done is not None:
                        on_cell_done()

            p1_tasks.append(_run_p1())
            p1_task_cids.append(cid)

        # Probe with the first call alone — if it fails immediately, abort before
        # launching hundreds of identical failing requests.
        if p1_tasks:
            try:
                first = await p1_tasks[0]
            except Exception as exc:
                logger.error("P1 online error on first call — aborting: %s", _format_llm_error(exc), exc_info=True)
                for coro in p1_tasks[1:]:
                    coro.close()
                return {}, {}, {}, _format_llm_error(exc)
            cid, text, usage = first
            p1_texts[cid] = text
            p1_usage[cid] = usage
            p1_tasks = p1_tasks[1:]
            p1_task_cids = p1_task_cids[1:]

        if p1_tasks:
            results_list = await asyncio.gather(*p1_tasks, return_exceptions=True)
            for task_cid, item in zip(p1_task_cids, results_list):
                if isinstance(item, Exception):
                    err_detail = _format_llm_error(item)
                    logger.error("P1 online error (cid=%s): %s", task_cid, err_detail)
                    p1_errors[task_cid] = err_detail
                else:
                    cid, text, usage = item
                    p1_texts[cid] = text
                    p1_usage[cid] = usage

    return p1_texts, p1_usage, p1_errors, None


async def _execute_pass1(
    run,
    cfg,
    papers,
    source_texts: dict,
    pending_cells: dict,
    store,
    settings,
    dry_run: bool,
    *,
    groups_def,
    sem: "asyncio.Semaphore | None" = None,
    limiter=None,
    on_cell_done: "Callable[[], None] | None" = None,
    on_total_known: "Callable[[int], None] | None" = None,
    build_p1_messages: "Callable[[str, list, int], list[dict]] | None" = None,
    should_skip_cell: "Callable[[int, int, int], bool] | None" = None,
) -> "tuple[dict, dict, dict, str | None]":
    """Run the Pass-1 phase.

    Returns ({custom_id: p1_text}, {custom_id: usage_stats}, {custom_id: error_detail}, fatal_error_or_None).

    ``pending_cells`` is mutated in-place: cleared then re-populated with
    ``{cid: (paper, group, group_idx)}`` for every cell that still needs work.
    Both batch and online sub-modes share this single build path; the caller sees
    the same map regardless of which sub-mode ran.

    ``groups_def`` (keyword-only) is the pre-resolved question groups for this run:
    either a single ``list[list[Question]]`` applied to every paper (obtained via
    ``resolve_groups(cfg, pipeline.questions)``), or a ``dict[paper_id, list[list[Question]]]``
    when groups vary per paper (e.g. arbitration's per-paper disputed questions).

    ``sem`` and ``limiter`` are optional concurrency/rate-limit guards applied to
    the online path only.  When both are None, behavior is unchanged.

    ``on_total_known`` is called once after pending_cells is built, with the total count.
    ``on_cell_done`` is called after each cell completes (success or failure).

    ``build_p1_messages``, when given, replaces the default annotation message
    builder for constructing Pass-1 messages: ``(source_text, group, paper_id) -> messages``.

    ``should_skip_cell``, when given, replaces ``store.should_skip_cell`` for the
    dedup/resume check: ``(run_id, paper_id, version_id) -> bool``.
    """
    return await _execute_pass1_with_groups(
        run, cfg, papers, source_texts, pending_cells, store, settings, dry_run, groups_def,
        sem=sem, limiter=limiter,
        on_cell_done=on_cell_done, on_total_known=on_total_known,
        build_p1_messages=build_p1_messages,
        should_skip_cell=should_skip_cell,
    )


async def _execute_pass2(
    run,
    cfg,
    pending_p1: dict,
    pending_cells: dict,
    store,
    settings,
    dry_run: bool,
    *,
    sem: "asyncio.Semaphore | None" = None,
    limiter=None,
    on_p2_start: "Callable[[int, str], None] | None" = None,
    on_p2_advance: "Callable[[], None] | None" = None,
) -> tuple[dict, dict, dict]:
    """Run the Pass-2 phase.

    Returns ({custom_id: p2_text}, {custom_id: usage_stats}, {custom_id: error_detail}).

    ``usage_stats`` carries the format-model (Pass-2) usage with keys
    ``input_tokens, output_tokens, cache_read_tokens, cost, latency_ms``.
    In batch mode, ``cost`` is computed after the fact from token counts (see
    ``_add_batch_costs``) and may be absent if that computation fails (e.g.
    unrecognized model); ``latency_ms`` is never available (the batch API does
    not expose per-item timing). Callers must read both via ``.get(...)`` with
    sensible defaults.

    ``error_detail`` is populated for cids that failed (batch item error, or an
    online call that raised); such cids are absent from the first two dicts.
    """
    from .config import DEFAULT_REQUEST_TIMEOUT, ProviderSettings
    from .annotate.prompt import build_format_messages
    from .annotate.parse import _RESPONSE_FORMAT
    from .llm import complete as llm_complete, dummy_complete

    _base_complete = dummy_complete if dry_run else llm_complete

    effective_fmt_provider = cfg.format_model_provider or run.model_provider
    effective_fmt_model = cfg.format_model or "gpt-4o-mini"
    fmt_prov_settings = settings.providers.get(effective_fmt_provider, ProviderSettings())
    p2_api_key = fmt_prov_settings.resolved_api_key()
    p2_base_url = fmt_prov_settings.base_url

    p2_response_format = _RESPONSE_FORMAT if cfg.format_structured_output else None
    batch_p2 = cfg.batch_p2

    if batch_p2 and not dry_run:
        p2_requests = []
        for cid, p1_text in pending_p1.items():
            paper, group, group_idx = pending_cells[cid]
            p2_messages = build_format_messages(p1_text, group)
            p2_requests.append(build_p2_request(
                custom_id=cid,
                provider=effective_fmt_provider,
                model=effective_fmt_model,
                messages=p2_messages,
                model_params=cfg.format_model_params,
                temperature=cfg.format_temperature,
                response_format=p2_response_format,
            ))

        if p2_requests:
            p2_provider = make_provider(effective_fmt_provider, p2_api_key, p2_base_url)
            p2_texts, p2_usage, p2_errors = await submit_and_poll(
                p2_provider, p2_requests, store, f"{run.run_id}:p2",
                label="Batch P2", pass_name="p2",
                provider_name=effective_fmt_provider, model=effective_fmt_model,
            )
        else:
            p2_texts = {}
            p2_usage = {}
            p2_errors = {}
    else:
        # P2 online — run concurrently
        p2_texts = {}
        p2_usage = {}
        p2_errors: dict[str, str] = {}
        if pending_p1:
            p2_label = f"Pass 2 ({effective_fmt_model})"

            # Use caller-supplied progress callbacks when available (avoids nested Live
            # contexts which cause terminal flickering); fall back to a self-owned Progress.
            if on_p2_start is not None:
                on_p2_start(len(pending_p1), p2_label)
                _advance_p2 = on_p2_advance or (lambda: None)
                _own_progress: "Progress | None" = None
            else:
                _own_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    console=_console,
                )
                _own_progress.start()
                _p2_prog = _own_progress.add_task(p2_label, total=len(pending_p1))
                _advance_p2 = lambda: _own_progress.advance(_p2_prog)  # type: ignore[union-attr]

            try:
                async def _run_p2(cid: str, p1_text: str, paper, group, group_idx: int):
                    p2_messages = build_format_messages(p1_text, group)
                    p2_kwargs: dict = {
                        "timeout": cfg.request_timeout or DEFAULT_REQUEST_TIMEOUT,
                    }
                    if cfg.format_temperature is not None:
                        p2_kwargs["temperature"] = cfg.format_temperature
                    p2_kwargs.update(cfg.format_model_params)
                    if p2_api_key:
                        p2_kwargs["api_key"] = p2_api_key
                    if p2_base_url:
                        p2_kwargs["api_base"] = p2_base_url
                    if p2_response_format is not None:
                        p2_kwargs["response_format"] = p2_response_format
                    if limiter is not None:
                        await limiter.acquire(effective_fmt_provider)
                    if sem is not None:
                        async with sem:
                            result = await _base_complete(
                                effective_fmt_model, effective_fmt_provider, p2_messages, **p2_kwargs
                            )
                    else:
                        result = await _base_complete(
                            effective_fmt_model, effective_fmt_provider, p2_messages, **p2_kwargs
                        )
                    _advance_p2()
                    usage = {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "cache_read_tokens": result.usage.cached_tokens,
                        "cost": result.cost,
                        "latency_ms": result.latency_ms,
                    }
                    return cid, result.text, usage

                p2_tasks = [
                    _run_p2(cid, p1_text, *pending_cells[cid])
                    for cid, p1_text in pending_p1.items()
                ]
                # Probe with the first call alone — abort if the first fails.
                p2_results: list = []
                if p2_tasks:
                    try:
                        p2_results.append(await p2_tasks[0])
                    except Exception as exc:
                        _console.print(
                            f"[bold red]LLM call failed[/] — {_format_llm_error(exc)}\n"
                            "[dim]Aborting remaining P2 calls.[/]"
                        )
                        logger.error("P2 online error on first call — aborting: %s", _format_llm_error(exc), exc_info=True)
                        for coro in p2_tasks[1:]:
                            coro.close()
                        return {}, {}, {}
                    if len(p2_tasks) > 1:
                        p2_results.extend(
                            await asyncio.gather(*p2_tasks[1:], return_exceptions=True)
                        )
            finally:
                if _own_progress is not None:
                    _own_progress.stop()

            p2_cids = list(pending_p1.keys())
            for cid, item in zip(p2_cids, p2_results):
                if isinstance(item, Exception):
                    err_detail = _format_llm_error(item)
                    logger.error("P2 online error (cid=%s): %s", cid, err_detail)
                    _console.print(f"[yellow]P2 error (skipping cell):[/] {err_detail}")
                    p2_errors[cid] = err_detail
                else:
                    got_cid, text, usage = item
                    p2_texts[got_cid] = text
                    p2_usage[got_cid] = usage

    return p2_texts, p2_usage, p2_errors
