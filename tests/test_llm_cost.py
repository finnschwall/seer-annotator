"""Cost is priced against the model we ASKED for, not the one echoed back.

Azure returns a dated snapshot name in the response body (`gpt-5.5-2026-04-24`)
that is not a key in litellm's price table, so pricing the echoed name alone
silently records no cost for every Azure call. See the comment in llm.py.

These tests drive the real litellm price table with a synthetic response — no
network — because the bug lives entirely in which name gets looked up.
"""

import asyncio
from decimal import Decimal

import litellm
import pytest
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

from seer_annotator.llm import complete


def _response(model: str, provider: str, *, cached: int = 0) -> ModelResponse:
    response = ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    )
    response.usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached),
    )
    response._hidden_params = {"custom_llm_provider": provider}
    return response


def _complete(monkeypatch, model: str, provider: str, response: ModelResponse):
    async def fake_acompletion(**kwargs):
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return asyncio.run(complete(model, provider, [{"role": "user", "content": "hi"}]))


def test_azure_snapshot_name_still_priced(monkeypatch):
    """The deployment name prices the call even though the echoed name is unmapped."""
    with pytest.raises(Exception):  # noqa: B017 - litellm raises a bare Exception
        litellm.completion_cost(
            completion_response=_response("gpt-5.5-2026-04-24", "azure")
        )

    result = _complete(
        monkeypatch, "gpt-5.5", "azure", _response("gpt-5.5-2026-04-24", "azure")
    )
    # 1000 input @ $5/M + 100 output @ $30/M
    assert result.cost == Decimal("0.008")


def test_azure_cache_reads_get_the_cheaper_rate(monkeypatch):
    result = _complete(
        monkeypatch,
        "gpt-5.5",
        "azure",
        _response("gpt-5.5-2026-04-24", "azure", cached=900),
    )
    # 100 uncached @ $5/M + 900 cached @ $0.50/M + 100 output @ $30/M
    assert result.cost == Decimal("0.00395")


def test_echoed_name_wins_when_litellm_knows_it(monkeypatch):
    """Anthropic echoes a name litellm prices; our fallback must not change it."""
    response = _response("claude-sonnet-5", "anthropic")
    expected = litellm.completion_cost(completion_response=response)

    result = _complete(monkeypatch, "claude-sonnet-5", "anthropic", response)
    assert result.cost == Decimal(str(expected))


def test_unpriced_endpoint_stays_unknown(monkeypatch):
    """A local OpenAI-compatible gateway has no price; None, never a fake $0."""
    result = _complete(
        monkeypatch,
        "kit.gemma4-31b-it",
        "openai",
        _response("kit.gemma4-31b-it", "openai"),
    )
    assert result.cost is None
