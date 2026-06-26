"""Async batch API execution for Pass-1 and/or Pass-2 requests.

Supports Anthropic (beta.messages.batches) and OpenAI (files + batches).
Google Gemini uses BigQuery/GCS and is architecturally different — not supported.

Each pass is independently toggleable via batch_p1 / batch_p2 in settings.toml [run_defaults] or per-run RunConfig.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from decimal import Decimal
from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status

logger = logging.getLogger(__name__)
_console = Console()

_DEFAULT_MAX_TOKENS_P1 = 4096
_DEFAULT_MAX_TOKENS_P2 = 1024
_POLL_START = 10    # seconds
_POLL_CAP = 120     # seconds


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BatchProvider(Protocol):
    def submit(self, requests: list[dict]) -> str: ...
    def poll(self, batch_id: str) -> str: ...  # "running" | "done" | "failed"
    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict]]:
        """Return ({cid: text}, {cid: usage_stats})."""
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

    def poll(self, batch_id: str) -> str:
        batch = self._client.beta.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return "done"
        if batch.processing_status == "canceling":
            return "failed"
        return "running"

    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict]]:
        results: dict[str, str] = {}
        usage_by_cid: dict[str, dict] = {}
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
                logger.warning("Batch item %s: result type=%s", cid, item.result.type)
                results[cid] = ""
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
        return results, usage_by_cid


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
        if batch.status in ("completed", "finalizing"):
            return "done"
        if batch.status in ("failed", "expired", "cancelled", "cancelling"):
            return "failed"
        return "running"

    def collect(self, batch_id: str) -> tuple[dict[str, str], dict[str, dict]]:
        batch = self._client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        if not output_file_id:
            logger.error("Batch %s has no output_file_id", batch_id)
            return {}, {}
        raw = self._client.files.content(output_file_id).read().decode()
        results: dict[str, str] = {}
        usage_by_cid: dict[str, dict] = {}
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
        return results, usage_by_cid


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
    temperature: float,
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
            "temperature": temperature,
            **extra,
        }
        if system:
            params["system"] = system
        return {"custom_id": custom_id, "params": params}

    # openai / azure
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra,
    }
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def build_p2_request(
    *,
    custom_id: str,
    provider: str,
    model: str,
    messages: list[dict],
    model_params: dict,
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
            "temperature": 0.0,
            **extra,
        }
        if system:
            params["system"] = system
        return {"custom_id": custom_id, "params": params}

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        **extra,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------

async def _poll_until_done(provider: BatchProvider, batch_id: str, label: str = "Batch") -> None:
    delay = _POLL_START
    t0 = time.monotonic()
    short_id = batch_id[:24] + ("…" if len(batch_id) > 24 else "")
    with Status(f"[bold]{label}[/] {short_id}", console=_console) as spinner:
        while True:
            result = provider.poll(batch_id)
            elapsed = int(time.monotonic() - t0)
            spinner.update(f"[bold]{label}[/] {short_id} [dim]{result} ({elapsed}s)[/]")
            logger.info("Batch %s: %s (%ds)", batch_id, result, elapsed)
            if result == "done":
                return
            if result == "failed":
                raise RuntimeError(f"Batch {batch_id} failed/expired/canceled")
            await asyncio.sleep(delay)
            delay = min(delay * 2, _POLL_CAP)


async def submit_and_poll(
    provider: BatchProvider,
    requests: list[dict],
    store,
    batch_key: str,
    label: str = "Batch",
) -> tuple[dict[str, str], dict[str, dict]]:
    """Submit a batch (or resume an existing one).

    Returns ({custom_id: text}, {custom_id: usage_stats}).
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

    await _poll_until_done(provider, batch_id, label=label)
    results, usage_by_cid = provider.collect(batch_id)
    store.delete_batch_id(batch_key)
    n_ok = len(results)
    _console.print(f"[bold]{label}[/] [green]✓[/] complete — {n_ok} result(s)")
    logger.info("Batch %s complete: %d results", batch_id, len(results))
    return results, usage_by_cid


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


async def run_batch_pipeline(
    pipeline,
    settings,
    *,
    store=None,
    client=None,
    dry_run: bool = False,
    run_ids: list[int] | None = None,
    paper_ids: list[int] | None = None,
) -> None:
    from .config import ProviderSettings, RunConfig, effective_run_config
    from .seer_client import SeerClient, DryRunSeerClient
    from .store import Store
    from .batching import resolve_groups
    from .caching import apply_cache
    from .annotate.prompt import build_messages, build_format_messages
    from .annotate.parse import ExtractionError, parse_structured_output
    from .annotate.verify import verify_citation
    from .mapping import build_llm_answer, build_error_answer
    from .llm import complete as llm_complete, dummy_complete

    pipeline_cfg = effective_run_config(RunConfig(), settings.run_defaults)

    import litellm as _litellm
    _litellm.drop_params = pipeline_cfg.drop_params

    store = store or Store(settings.runtime.store_path)
    client = client or (
        DryRunSeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
        if dry_run
        else SeerClient(pipeline.api_base, pipeline.api_token, pipeline.review_id, pipeline.questions)
    )

    _base_complete = dummy_complete if dry_run else llm_complete
    runs = [r for r in pipeline.runs if (run_ids is None or r.run_id in run_ids)]
    papers = [p for p in pipeline.papers if (paper_ids is None or p.paper_id in paper_ids)]

    for run in runs:
        cfg = effective_run_config(run.config, settings.run_defaults)
        batch_p1 = cfg.batch_p1
        batch_p2 = cfg.batch_p2
        prov_settings = settings.providers.get(run.model_provider, ProviderSettings())
        p1_api_key = prov_settings.resolved_api_key()
        p1_base_url = prov_settings.base_url

        effective_fmt_provider = cfg.format_model_provider or run.model_provider
        effective_fmt_model = cfg.format_model or "gpt-4o-mini"
        fmt_prov_settings = settings.providers.get(effective_fmt_provider, ProviderSettings())
        p2_api_key = fmt_prov_settings.resolved_api_key()
        p2_base_url = fmt_prov_settings.base_url

        # Validate that batch providers are supported before starting
        if batch_p1 and not dry_run:
            make_provider(run.model_provider, p1_api_key, p1_base_url)  # raises on unsupported
        if batch_p2 and not dry_run:
            make_provider(effective_fmt_provider, p2_api_key, p2_base_url)

        groups_def = resolve_groups(cfg, pipeline.questions)

        # ---- Collect source texts ----
        source_texts: dict[int, str] = {}
        for paper in papers:
            if cfg.text_source == "full_text":
                source = store.get_ocr(paper.paper_id)
                if source is None:
                    source = await client.fetch_ocr_markdown(paper.paper_id)
                    store.save_ocr(paper.paper_id, source)
                if source is None:
                    logger.warning("No OCR for paper %d — posting error for all questions", paper.paper_id)
                    for q in pipeline.questions:
                        store.save_answer(
                            run.run_id, paper.paper_id, q.version_id,
                            build_error_answer(run_id=run.run_id, paper_id=paper.paper_id, question=q, extraction_detail="no_ocr"),
                        )
                    continue
                source_texts[paper.paper_id] = source
            else:
                source_texts[paper.paper_id] = paper.abstract

        effective_system = cfg.system_prompt

        # ---- Phase 1 ----
        # p1_texts: {custom_id: text}
        if batch_p1 and not dry_run:
            # Build the full pending_cells map first (needed regardless of dump)
            pending_cells: dict[str, tuple] = {}
            for paper in papers:
                if paper.paper_id not in source_texts:
                    continue
                source = source_texts[paper.paper_id]
                for group_idx, group in enumerate(groups_def):
                    cells = [(run.run_id, paper.paper_id, q.version_id) for q in group]
                    if all(store.should_skip_cell(*c) for c in cells):
                        continue
                    cid = f"{run.run_id}-{paper.paper_id}-{group_idx}"
                    pending_cells[cid] = (paper, group, group_idx)

            # Try loading a saved P1 dump (crash-safe resume without re-billing)
            p1_texts: dict[str, str] = _load_p1_dump(settings.runtime.p1_dump_dir, run.run_id) or {}
            p1_usage: dict[str, dict] = {}  # {cid: usage_stats} — empty when loaded from dump

            if not p1_texts:
                if not pending_cells:
                    logger.info("Run %d: no pending P1 work", run.run_id)
                else:
                    p1_requests = []
                    for cid, (paper, group, _) in pending_cells.items():
                        source = source_texts[paper.paper_id]
                        messages = build_messages(
                            source,
                            group,
                            text_source=cfg.text_source,
                            system_prompt=effective_system,
                            cache_first=cfg.cache_first,
                        )
                        messages = apply_cache(run.model_provider, messages, cfg.cache, ttl=cfg.cache_ttl)
                        p1_requests.append(build_p1_request(
                            custom_id=cid,
                            provider=run.model_provider,
                            model=run.model_name,
                            messages=messages,
                            temperature=cfg.temperature,
                            model_params=cfg.model_params,
                        ))

                    provider = make_provider(run.model_provider, p1_api_key, p1_base_url)

                    # Cache pre-warm: submit the first request alone so it writes
                    # the shared prefix to the 1h cache before the main batch starts.
                    # All subsequent requests then hit the warm cache instead of racing
                    # to write it in parallel (which caused ~48 writes vs 1 in testing).
                    # Only worthwhile when caching is enabled and there are multiple requests.
                    use_prewarm = cfg.cache and len(p1_requests) > 1
                    if use_prewarm:
                        logger.info(
                            "Run %d: submitting prewarm request to warm 1h cache before main batch",
                            run.run_id,
                        )
                        prewarm_texts, prewarm_usage = await submit_and_poll(
                            provider, p1_requests[:1], store, f"{run.run_id}:p1_prewarm",
                            label="Cache warm-up",
                        )
                        main_texts, main_usage = await submit_and_poll(
                            provider, p1_requests[1:], store, f"{run.run_id}:p1",
                            label="Batch P1",
                        )
                        p1_texts = {**prewarm_texts, **main_texts}
                        p1_usage = {**prewarm_usage, **main_usage}
                    else:
                        p1_texts, p1_usage = await submit_and_poll(
                            provider, p1_requests, store, f"{run.run_id}:p1",
                            label="Batch P1",
                        )

                    _save_p1_dump(settings.runtime.p1_dump_dir, run.run_id, p1_texts)
        else:
            # P1 online: gather all papers concurrently
            p1_texts = {}
            p1_usage = {}
            pending_cells = {}
            p1_tasks = []

            for paper in papers:
                if paper.paper_id not in source_texts:
                    continue
                source = source_texts[paper.paper_id]
                for group_idx, group in enumerate(groups_def):
                    cells = [(run.run_id, paper.paper_id, q.version_id) for q in group]
                    if all(store.should_skip_cell(*c) for c in cells):
                        continue
                    cid = f"{run.run_id}-{paper.paper_id}-{group_idx}"
                    pending_cells[cid] = (paper, group, group_idx)

                    async def _run_p1(cid=cid, paper=paper, group=group, source=source):
                        messages = build_messages(
                            source,
                            group,
                            text_source=cfg.text_source,
                            system_prompt=effective_system,
                            cache_first=cfg.cache_first,
                        )
                        messages = apply_cache(run.model_provider, messages, cfg.cache, ttl=cfg.cache_ttl)
                        p1_kwargs: dict = {"temperature": cfg.temperature}
                        if cfg.reasoning_effort:
                            p1_kwargs["reasoning_effort"] = cfg.reasoning_effort
                        p1_kwargs.update(cfg.model_params)
                        if p1_api_key:
                            p1_kwargs["api_key"] = p1_api_key
                        if p1_base_url:
                            p1_kwargs["api_base"] = p1_base_url
                        result = await _base_complete(
                            run.model_name, run.model_provider, messages, **p1_kwargs
                        )
                        return cid, result.text

                    p1_tasks.append(_run_p1())

            results_list = await asyncio.gather(*p1_tasks, return_exceptions=True)
            for item in results_list:
                if isinstance(item, Exception):
                    logger.error("P1 online error: %s", item)
                    continue
                cid, text = item
                p1_texts[cid] = text

        # ---- Phase 2 ----
        # Only process entries that are still pending (dump may contain already-completed cells)
        pending_p1 = {cid: text for cid, text in p1_texts.items() if cid in pending_cells}

        from .annotate.engine import _RESPONSE_FORMAT
        p2_response_format = _RESPONSE_FORMAT if cfg.format_structured_output else None

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
                    response_format=p2_response_format,
                ))

            if p2_requests:
                p2_provider = make_provider(effective_fmt_provider, p2_api_key, p2_base_url)
                p2_texts, _ = await submit_and_poll(
                    p2_provider, p2_requests, store, f"{run.run_id}:p2",
                    label="Batch P2",
                )
            else:
                p2_texts = {}
        else:
            # P2 online — run concurrently with a progress bar
            p2_texts = {}
            if pending_p1:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    console=_console,
                ) as progress:
                    p2_prog = progress.add_task(
                        f"Pass 2 ({effective_fmt_model})", total=len(pending_p1)
                    )

                    async def _run_p2(cid: str, p1_text: str, paper, group, group_idx: int):
                        p2_messages = build_format_messages(p1_text, group)
                        p2_kwargs: dict = {"temperature": 0.0}
                        p2_kwargs.update(cfg.format_model_params)
                        if p2_api_key:
                            p2_kwargs["api_key"] = p2_api_key
                        if p2_base_url:
                            p2_kwargs["api_base"] = p2_base_url
                        if p2_response_format is not None:
                            p2_kwargs["response_format"] = p2_response_format
                        result = await _base_complete(
                            effective_fmt_model, effective_fmt_provider, p2_messages, **p2_kwargs
                        )
                        progress.advance(p2_prog)
                        return cid, result.text

                    p2_tasks = [
                        _run_p2(cid, p1_text, *pending_cells[cid])
                        for cid, p1_text in pending_p1.items()
                    ]
                    p2_results = await asyncio.gather(*p2_tasks, return_exceptions=True)

                for item in p2_results:
                    if isinstance(item, Exception):
                        logger.error("P2 online error: %s", item)
                        continue
                    cid, text = item
                    p2_texts[cid] = text

        # ---- Parse, save, post ----
        first_extraction_error: ExtractionError | None = None

        for cid in p2_texts:
            if cid not in pending_cells:
                continue
            paper, group, group_idx = pending_cells[cid]
            p1_text = p1_texts.get(cid, "")
            p2_text = p2_texts[cid]

            u = p1_usage.get(cid, {})
            tok_input  = u.get("input_tokens", 0)
            tok_output = u.get("output_tokens", 0)
            tok_cached = u.get("cache_read_tokens", 0)

            parsed = parse_structured_output(p2_text, [q.key for q in group])

            failed_keys = {r["key"]: r["parse_error"] for r in parsed if "parse_error" in r}
            if failed_keys:
                err = ExtractionError(failed_keys)
                logger.error(
                    "EXTRACTION FAILURE — run %d, paper %d, group %d: %s\n"
                    "  Pass-2 output was:\n%s",
                    run.run_id, paper.paper_id, group_idx, err,
                    p2_text[:2000],
                )
                for q in group:
                    store.save_answer(
                        run.run_id, paper.paper_id, q.version_id,
                        build_error_answer(run_id=run.run_id, paper_id=paper.paper_id, question=q, extraction_detail=str(err)),
                    )
                if cfg.fail_fast and first_extraction_error is None:
                    first_extraction_error = err
                continue

            for i, (question, result) in enumerate(zip(group, parsed)):
                verify = verify_citation(
                    result.get("cited_text", ""),
                    source_texts.get(paper.paper_id, ""),
                    max_error_rate=cfg.citation_max_error_rate,
                    max_ellipsis_gap=cfg.citation_max_ellipsis_gap,
                )
                cited_text_verified = None if verify.get("note") == "no citation provided" else verify["ok"]
                raw_response = {
                    "pass1_text": p1_text,
                    "pass2_text": p2_text,
                    "parse_result": result,
                    "verify": verify,
                    "text_source": cfg.text_source,
                    "batch_group_id": cid,
                    "batch_mode": True,
                    "p1_usage": u,
                }
                payload = build_llm_answer(
                    run_id=run.run_id,
                    paper_id=paper.paper_id,
                    question=question,
                    value=result.get("value"),
                    comment=result.get("comment", ""),
                    cited_text=result.get("cited_text", ""),
                    cited_text_verified=cited_text_verified,
                    raw_response=raw_response,
                    latency_ms=0,
                    tokens_total=tok_input + tok_output + tok_cached,
                    tokens_input=tok_input,
                    tokens_output=tok_output,
                    tokens_cached=tok_cached,
                    cost=None,
                    cost_currency="USD",
                    confidence=result.get("confidence"),
                )
                store.save_answer(run.run_id, paper.paper_id, question.version_id, payload, cid)

        if first_extraction_error is not None and cfg.fail_fast:
            raise first_extraction_error

        # Post per-paper (includes no-OCR error records)
        for paper in papers:
            unposted = store.get_unposted(run.run_id, paper.paper_id)
            if not unposted:
                continue
            try:
                await client.post_answers_bulk(unposted)
                if not dry_run:
                    version_ids = [p["question_version"] for p in unposted]
                    store.mark_posted(run.run_id, paper.paper_id, version_ids)
                logger.info("Posted %d answers for paper %d", len(unposted), paper.paper_id)
            except Exception as exc:
                logger.error("Post failed for paper %d: %s", paper.paper_id, exc)

        logger.info("Run %d batch pipeline complete", run.run_id)
