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

    # Cost via LiteLLM
    cost_val: Decimal | None = None
    try:
        cost_float = litellm.completion_cost(completion_response=response)
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
        text = (
            "STUDY_DESIGN_KEY\n"
            "Quote: 'patients were randomised'\n"
            "Reasoning: The paper describes randomisation.\n"
            "Answer: rct\n"
            "Confidence: 4\n"
        )
    return _DummyResult(text=text, model=model)  # type: ignore[return-value]
