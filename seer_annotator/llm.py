"""LiteLLM wrapper returning normalized text/usage/cost/raw."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    reasoning_content: str | None
    usage: UsageInfo
    cost: Decimal | None
    cost_currency: str
    latency_ms: int
    raw: dict = field(default_factory=dict)
    # Why the model stopped. "length" means it hit max_tokens with more to say,
    # i.e. `text` is cut off mid-sentence — the one finish reason a caller must
    # not treat as a complete answer. None when the provider reported nothing.
    # See TRUNCATED_FINISH_REASONS in batch_runner.py.
    finish_reason: str | None = None


async def complete(
    model: str,
    provider: str,
    messages: list[dict],
    *,
    response_format: dict | None = None,
    **params: Any,
) -> LLMResult:
    import litellm  # lazy import so tests can stub it

    # LiteLLM model string: always "provider/model" so LiteLLM knows the provider
    # even when api_base is overridden (e.g. local OpenAI-compatible endpoints)
    litellm_model = f"{provider}/{model}"

    call_kwargs: dict[str, Any] = dict(
        model=litellm_model,
        messages=messages,
        **params,
    )
    if response_format is not None:
        call_kwargs["response_format"] = response_format

    t0 = time.monotonic()
    response = await litellm.acompletion(**call_kwargs)
    latency_ms = int((time.monotonic() - t0) * 1000)

    choice = response.choices[0]
    text: str = choice.message.content or ""
    reasoning: str | None = getattr(choice.message, "reasoning_content", None)
    finish_reason: str | None = getattr(choice, "finish_reason", None)

    # Normalize usage
    u = response.usage or {}
    raw_usage = u if isinstance(u, dict) else (u.model_dump() if hasattr(u, "model_dump") else vars(u))

    ptd = raw_usage.get("prompt_tokens_details") or {}
    ctd = raw_usage.get("completion_tokens_details") or {}

    usage = UsageInfo(
        input_tokens=raw_usage.get("prompt_tokens", 0) or 0,
        output_tokens=raw_usage.get("completion_tokens", 0) or 0,
        cached_tokens=(ptd.get("cached_tokens") or 0),
        reasoning_tokens=(ctd.get("reasoning_tokens") or 0),
        total_tokens=raw_usage.get("total_tokens", 0) or 0,
    )

    # Ollama reports no thinking-token count: its `/api/chat` response carries only
    # `prompt_eval_count` and `eval_count`, and thinking is folded into the latter.
    # It does return the thinking TEXT, so split the reported completion tokens by
    # character share -- both strings come from the same tokenizer, so their
    # chars-per-token cancels. Measured 0-6% against `eval_count - tokens(content)`
    # on gemma4:31b-cloud and glm-5.2:cloud (worst case: long visible answers).
    #
    # Deliberately keyed on "text but no count" rather than on a provider name, so
    # it cannot fire for a provider that reports a real number. It never applies to
    # Anthropic: with `thinking.display` at its default ("omitted" on Sonnet 5 /
    # Opus 5 / Opus 4.7+) there is no text to measure, and the count is reported
    # anyway. Counts derived here are approximate -- callers that display them
    # should mark them as such.
    if not usage.reasoning_tokens and reasoning and usage.output_tokens:
        share = len(reasoning) / (len(reasoning) + len(text))
        usage.reasoning_tokens = round(share * usage.output_tokens)

    # Cost via LiteLLM.
    #
    # `model=litellm_model` is required, not decoration. Left to itself,
    # completion_cost() prices the name the PROVIDER echoed back in the response
    # body, and Azure echoes a dated snapshot rather than the deployment you
    # called: ask for `azure/gpt-5.5` and the response says
    # `gpt-5.5-2026-04-24`, which is not a key in litellm's price table (it has
    # `azure/gpt-5.5-2026-04-23`, one day off). completion_cost() then raises
    # "This model isn't mapped yet", the except below swallows it, and every
    # Azure call is recorded with no cost at all.
    #
    # Passing our own model name adds it as a second candidate: the response's
    # name is still tried first, so a provider that echoes something litellm
    # knows is unaffected (measured: identical results for anthropic and
    # ollama). It is only a fallback, so a genuinely unpriced endpoint
    # (hosted_vllm, an OpenAI-compatible local gateway) still ends up as None —
    # "unknown", which is the honest answer, and not a misleading $0.
    cost_val: Decimal | None = None
    try:
        cost_float = litellm.completion_cost(
            completion_response=response, model=litellm_model
        )
        if cost_float is not None:
            cost_val = Decimal(str(cost_float))
    except Exception:
        pass

    raw_dict: dict = {}
    try:
        raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
    except Exception:
        pass

    return LLMResult(
        text=text,
        reasoning_content=reasoning,
        usage=usage,
        cost=cost_val,
        cost_currency="USD",
        latency_ms=latency_ms,
        raw=raw_dict,
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Dummy provider for tests / dry-run
# ---------------------------------------------------------------------------

class _DummyResult:
    """Returned by dummy_complete; mimics LLMResult."""

    def __init__(self, text: str, model: str) -> None:
        self.text = text
        self.reasoning_content = None
        self.usage = UsageInfo(
            input_tokens=10, output_tokens=5, cached_tokens=0, reasoning_tokens=0, total_tokens=15
        )
        self.cost = Decimal("0.0")
        self.cost_currency = "USD"
        self.latency_ms = 1
        self.raw = {"dummy": True, "model": model}
        self.finish_reason = "stop"


def _dummy_question_keys(messages: list[dict]) -> list[str]:
    """Pull question keys out of a build_messages()-shaped prompt, from its
    ``--- QUESTION: <key> ---`` section headers (see annotate/prompt.py)."""
    import re

    content = " ".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    )
    return re.findall(r"--- QUESTION: (\S+) ---", content)


async def dummy_complete(
    model: str,
    provider: str,
    messages: list[dict],
    *,
    response_format: dict | None = None,
    **params: Any,
) -> LLMResult:  # type: ignore[return-value]
    if response_format is not None:
        import json
        import re
        content = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        keys = re.findall(r"key='([^']+)'", content)
        results = [
            {"key": k, "value": None, "cited_text": "", "comment": "dummy", "confidence": None}
            for k in keys
        ]
        text = json.dumps({"results": results})
    else:
        # Emit a real "--- ANSWER: <key> ---" block per question found in the
        # prompt (see annotate/prompt.py's DEFAULT_SYSTEM template) so this
        # dummy stands in faithfully for a Pass-1 model under the
        # pass1_block_present() presence check in annotate/scope.py — a
        # canned, key-agnostic block would otherwise get every "ok" demoted
        # to "absent" as a false-positive fabrication.
        keys = _dummy_question_keys(messages)
        if keys:
            text = "\n".join(
                f"--- ANSWER: {k} ---\n"
                "Quotes:\n"
                "- \"patients were randomised\"\n"
                "Reasoning: The paper describes randomisation.\n"
                "Answer: rct\n"
                "Confidence: 4\n"
                for k in keys
            )
        else:
            # No recognizable question blocks (e.g. a caller passing custom
            # messages directly, such as arbitration) — fall back to the
            # original generic text.
            text = (
                "STUDY_DESIGN_KEY\n"
                "Quote: 'patients were randomised'\n"
                "Reasoning: The paper describes randomisation.\n"
                "Answer: rct\n"
                "Confidence: 4\n"
            )
    return _DummyResult(text=text, model=model)  # type: ignore[return-value]
