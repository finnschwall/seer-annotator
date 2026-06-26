"""Provider-aware cache marker placement for the paper-prefix message block."""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

_GEMINI_MIN_TOKENS = 1024  # skip if prefix is likely shorter


def apply_cache(
    provider: str,
    messages: list[dict],
    enabled: bool,
    ttl: str = "1h",
) -> list[dict]:
    """Return a (possibly new) messages list with cache markers applied.

    Only the paper-prefix block (first user message) gets annotated.
    `ttl` is only used for Anthropic; valid values are "5m" and "1h".
    """
    if not enabled:
        return messages

    provider = provider.lower()

    if provider in ("anthropic",):
        return _mark_anthropic(messages, ttl=ttl)

    if provider in ("google", "gemini", "vertex_ai"):
        return _mark_gemini(messages)

    # openai, local, vllm, sglang: prefix caching is automatic / server-side
    return messages


def _mark_anthropic(messages: list[dict], ttl: str = "1h") -> list[dict]:
    """Add cache_control to the content of the first user message.

    Uses the 1-hour TTL by default so a pre-warm request written before a batch
    stays alive for the full batch run (batches often exceed 5 minutes).
    Pass ttl="5m" for the cheaper 5-minute TTL in non-batch contexts.
    """
    messages = copy.deepcopy(messages)
    cc = {"type": "ephemeral", "ttl": ttl}
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content, "cache_control": cc}]
            elif isinstance(content, list) and content:
                # Mark only the last content block (Anthropic recommendation)
                last = copy.deepcopy(content[-1])
                last["cache_control"] = cc
                msg["content"] = list(content[:-1]) + [last]
            return messages  # only first user message
    return messages


def _mark_gemini(messages: list[dict]) -> list[dict]:
    """Mark first user message for Gemini, only if it looks long enough."""
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            text = content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            # Rough token estimate: 1 token ≈ 4 chars
            if len(text) < _GEMINI_MIN_TOKENS * 4:
                logger.debug("Gemini: prefix too short for caching, skipping")
                return messages  # don't mark

            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                last = copy.deepcopy(content[-1])
                last["cache_control"] = {"type": "ephemeral"}
                msg["content"] = list(content[:-1]) + [last]
            return messages
    return messages


def extract_cache_tokens(raw_usage: dict, provider: str) -> dict[str, int]:
    """Return {cached_read, cached_write} from raw LiteLLM usage dict."""
    provider = provider.lower()
    if provider == "anthropic":
        return {
            "cached_read": raw_usage.get("cache_read_input_tokens", 0) or 0,
            "cached_write": raw_usage.get("cache_creation_input_tokens", 0) or 0,
        }
    # openai / gemini / local: cached is under prompt_tokens_details
    ptd = raw_usage.get("prompt_tokens_details") or {}
    return {
        "cached_read": ptd.get("cached_tokens", 0) or 0,
        "cached_write": 0,
    }
