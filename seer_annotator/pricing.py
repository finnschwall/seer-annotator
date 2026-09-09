"""Resolving a model name to a price, including through a gateway.

A deployment often reaches a model through something that is not the model's
own vendor: an Open WebUI instance, a LiteLLM proxy, a university or corporate
gateway. Those namespace their models -- ``google.gemini-3.5-flash``,
``azure.gpt-5.5``, ``kit.glm-5.3`` -- and they are reached over the OpenAI
dialect, so the name litellm is asked to price is ``openai/google.gemini-3.5-flash``,
which is in no price table anywhere. Every such call used to record no cost at
all.

So the lookup walks four candidates, and stops at the first one litellm knows:

    1. "<provider>/<model>"   the name we actually called
    2. "<model>"              litellm keys some vendors bare (claude-sonnet-5)
    3. "<model minus the leading vendor. prefix>"
    4. the same, with dots turned into dashes  (claude-haiku-4.5 -> claude-haiku-4-5,
       because litellm spells Gemini with dots and Anthropic with dashes)

Steps 1 and 2 are the name the deployment asked for, so a price found there is
the vendor's list price for the model that was actually called: `inferred` is
False. Steps 3 and 4 are a *guess* at which underlying model sits behind a
gateway's label, so `inferred` is True and every surface that prints the number
has to say so -- see `experiments/price_basis.py` on the SEER side.

That distinction is the whole point of this module, and it is not pedantry. The
guess is right for a gateway that passes through to the real vendor
(``google.gemini-3.5-flash`` really is Gemini, billed by Google) and wrong for
one that self-hosts under a vendor's model name (``kit.deepseek-v4-flash`` runs
on the university's own GPUs, where DeepSeek's API rate is meaningless). No
rule can tell those apart from the name alone -- only the person reading the
number can -- so the number is shown with its basis rather than withheld or
asserted.

What is NOT done: registering guessed prices into `litellm.model_cost`. That
would make the guess indistinguishable from a real entry at every later lookup,
including ones in litellm's own internals, and there would be nothing left to
label.
"""

from __future__ import annotations

import io
import contextlib
import logging
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceBasis:
    """Which price table entry a cost came from, and whether it was a guess.

    ``key`` is the litellm model key that carried the rates. ``inferred`` is
    True when the key was reached by stripping a gateway's namespace prefix,
    i.e. we decided which model this probably is.
    """

    key: str
    inferred: bool


def _namespace_tail(model: str) -> str | None:
    """The part of ``model`` after a leading ``<vendor>.`` namespace, if any.

    Only a prefix that looks like a vendor word counts. A version number is not
    a namespace, so ``gpt-4.1`` is left alone -- its prefix ``gpt-4`` contains a
    digit, which no vendor namespace does.
    """
    prefix, dot, tail = model.partition(".")
    if not dot or not tail or not prefix:
        return None
    if any(c.isdigit() for c in prefix):
        return None
    return tail


def price_key_candidates(provider: str, model: str) -> list[tuple[str, bool]]:
    """The lookup ladder for ``provider``/``model``, as (key, inferred) pairs.

    Ordered most-specific first. Duplicates are dropped so a bare-name provider
    is not probed twice.
    """
    ladder: list[tuple[str, bool]] = [
        (f"{provider}/{model}", False),
        (model, False),
    ]
    tail = _namespace_tail(model)
    if tail:
        ladder.append((tail, True))
        if "." in tail:
            ladder.append((tail.replace(".", "-"), True))

    seen: set[str] = set()
    unique: list[tuple[str, bool]] = []
    for key, inferred in ladder:
        if key not in seen:
            seen.add(key)
            unique.append((key, inferred))
    return unique


def _lookup(key: str) -> dict | None:
    """litellm's price entry for ``key``, or None. Never raises, never prints.

    litellm signals "no such model" by raising, and writes a provider list to
    stdout on the way out -- which would otherwise appear in the middle of a
    run's log for every unpriced local model.
    """
    import litellm

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            info = litellm.get_model_info(model=key)
    except Exception:
        return None
    if not info:
        return None
    # An entry with no rates at all prices nothing; treat it as absent so the
    # ladder keeps walking rather than settling on a useless hit.
    if info.get("input_cost_per_token") is None and info.get("output_cost_per_token") is None:
        return None
    return info


@lru_cache(maxsize=512)
def model_info(provider: str, model: str) -> tuple[dict | None, PriceBasis | None]:
    """litellm's price entry for a model, plus where it came from.

    ``(None, None)`` means no candidate resolved -- an unpriced endpoint, which
    is the honest answer and not a misleading $0.
    """
    for key, inferred in price_key_candidates(provider, model):
        info = _lookup(key)
        if info is not None:
            if inferred:
                logger.info(
                    "No price entry for %s/%s — using %s (inferred from the gateway prefix)",
                    provider, model, key,
                )
            return info, PriceBasis(key=key, inferred=inferred)
    return None, None


def resolve(provider: str, model: str) -> PriceBasis | None:
    """Where a price for ``provider``/``model`` would come from, or None."""
    return model_info(provider, model)[1]


def cost_from_usage(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> tuple[Decimal | None, PriceBasis | None]:
    """Price one completion from its token counts, for a model litellm cannot price itself.

    `litellm.completion_cost()` cannot be used for this. It reads the provider
    off the response's own `_hidden_params`, which for a gateway is always
    `openai`, and no `model=` or `custom_llm_provider=` argument overrides it —
    so a gateway model is looked up as `openai/<whatever you pass>` and misses
    however good the name is. Measured against the live endpoint: every one of
    the four ladder candidates raised "This model isn't mapped yet".

    So the rates are applied here instead. This is the fallback path only:
    `llm.py` still asks `completion_cost` first, which keeps every provider it
    already handles (and its Azure dated-snapshot behaviour) exactly as it was.

    Input tokens are assumed to INCLUDE the cached ones, which is what the
    OpenAI dialect reports — and a gateway, by definition, speaks it. Anthropic
    never reaches this path: litellm prices it at step 1.
    """
    from decimal import Decimal

    info, basis = model_info(provider, model)
    if not info:
        return None, None

    def rate(*keys: str) -> Decimal:
        for key in keys:
            value = info.get(key)
            if value:
                return Decimal(str(value))
        return Decimal(0)

    input_rate = rate("input_cost_per_token")
    output_rate = rate("output_cost_per_token")
    if not input_rate and not output_rate:
        return None, None
    # A provider that prices no cache class bills those tokens as ordinary input.
    cache_read_rate = rate("cache_read_input_token_cost", "input_cost_per_token")

    cached = max(0, cached_tokens or 0)
    uncached = max(0, (input_tokens or 0) - cached)
    total = (
        uncached * input_rate
        + cached * cache_read_rate
        + (output_tokens or 0) * output_rate
    )
    return total, basis
