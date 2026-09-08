"""Tests for the observable submit→poll batch redesign (batch_runner.py).

Covers: single-shot polling (BatchPendingError raised on non-terminal status,
no sleep loop), kv cleared on failure but NOT on pending, and the 3-tuple
collect()/submit_and_poll() return shape (results, usage_by_cid, errors_by_cid)
with failed cids excluded from `results` rather than sentineled as "".
"""

import copy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from seer_annotator.batch_runner import (
    TRUNCATED_FINISH_REASONS,
    AnthropicBatchProvider,
    BatchPendingError,
    _batch_item_cost,
    _demote_truncated_p1,
    _execute_pass2,
    _is_transient_transport_error,
    _poll_once,
    build_p1_request,
    submit_and_poll,
)
from seer_annotator.orchestrator import _p1_drop_reason
from seer_annotator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


class _FakeProvider:
    """Minimal BatchProvider stub — records calls, returns scripted responses."""

    def __init__(self, poll_sequence, collect_result=None):
        self._poll_sequence = list(poll_sequence)
        self._collect_result = collect_result
        self.submit_calls = 0
        self.poll_calls = 0
        self.collect_calls = 0

    def submit(self, requests):
        self.submit_calls += 1
        return "batch-123"

    def poll(self, batch_id):
        self.poll_calls += 1
        return self._poll_sequence.pop(0)

    def collect(self, batch_id):
        self.collect_calls += 1
        return self._collect_result


# ---------------------------------------------------------------------------
# _poll_once
# ---------------------------------------------------------------------------

def test_poll_once_done_returns_and_keeps_no_state(store):
    provider = _FakeProvider(["done"])
    store.save_batch_id("k", "batch-123")
    _poll_once(provider, "batch-123", "k", store, label="Batch P1")
    assert provider.poll_calls == 1


def test_poll_once_running_raises_pending_and_keeps_kv(store):
    provider = _FakeProvider(["running"])
    store.save_batch_id("k", "batch-123")
    with pytest.raises(BatchPendingError) as exc_info:
        _poll_once(provider, "batch-123", "k", store, label="Batch P1", pass_name="p1", request_count=7)
    err = exc_info.value
    assert err.batch_id == "batch-123"
    assert err.status == "running"
    assert err.pass_name == "p1"
    assert err.request_count == 7
    # kv must be left in place so a later external call resumes the same batch_id.
    assert store.get_batch_id("k") == "batch-123"
    assert provider.poll_calls == 1  # single-shot: exactly one poll, no loop


def test_poll_once_failed_raises_runtime_error_and_clears_kv(store):
    provider = _FakeProvider(["failed"])
    store.save_batch_id("k", "batch-123")
    with pytest.raises(RuntimeError):
        _poll_once(provider, "batch-123", "k", store, label="Batch P1")
    # Previously only cleared on success — a failed batch would be re-polled
    # forever on retry. Must be cleared now.
    assert store.get_batch_id("k") is None


# ---------------------------------------------------------------------------
# owes_batch — was this batch collected?
# ---------------------------------------------------------------------------

def test_owes_batch_tracks_the_whole_submit_collect_lifecycle(store):
    """The store is the only durable record of whether results were taken."""
    assert not store.owes_batch("batch-123")
    store.save_batch_id("run1:p1", "batch-123")
    assert store.owes_batch("batch-123")
    assert not store.owes_batch("some-other-batch")
    store.delete_batch_id("run1:p1")
    assert not store.owes_batch("batch-123")


@pytest.mark.asyncio
async def test_submit_and_poll_clears_the_debt_only_after_collect(store):
    provider = _FakeProvider(["running"])
    with pytest.raises(BatchPendingError):
        await submit_and_poll(provider, [{"custom_id": "c1"}], store, "run1:p1", label="Batch P1")
    assert store.owes_batch("batch-123")

    provider2 = _FakeProvider(["done"], collect_result=({"c1": "hi"}, {}, {}))
    await submit_and_poll(provider2, [{"custom_id": "c1"}], store, "run1:p1", label="Batch P1")
    assert not store.owes_batch("batch-123")


# ---------------------------------------------------------------------------
# Unreachable provider on a read call — "don't know yet", not "failed"
# ---------------------------------------------------------------------------

class _UnreachableProvider(_FakeProvider):
    """Fails `poll` (or `collect`) with a given exception instead of answering."""

    def __init__(self, exc, *, on="poll", poll_sequence=("done",), collect_result=None):
        super().__init__(poll_sequence, collect_result=collect_result)
        self._exc = exc
        self._on = on

    def poll(self, batch_id):
        self.poll_calls += 1
        if self._on == "poll":
            raise self._exc
        return self._poll_sequence.pop(0)

    def collect(self, batch_id):
        self.collect_calls += 1
        if self._on == "collect":
            raise self._exc
        return self._collect_result


def _timeout_error():
    """An `anthropic.APITimeoutError`, exactly as raised on a TLS handshake timeout."""
    import anthropic
    import httpx
    return anthropic.APITimeoutError(request=httpx.Request("GET", "https://api.anthropic.com/x"))


def test_is_transient_recognises_sdk_timeout_and_5xx_and_rate_limit():
    import httpx
    assert _is_transient_transport_error(_timeout_error())
    assert _is_transient_transport_error(httpx.ConnectError("connection refused"))
    assert _is_transient_transport_error(TimeoutError("timed out"))
    # Wrapped one layer down — SDKs re-raise their own class over the socket error.
    wrapped = RuntimeError("boom")
    wrapped.__cause__ = httpx.ReadTimeout("slow")
    assert _is_transient_transport_error(wrapped)
    # A definite answer from the provider is not a blip.
    assert not _is_transient_transport_error(ValueError("bad json"))
    assert not _is_transient_transport_error(_StatusError(404))
    assert _is_transient_transport_error(_StatusError(503))


class _StatusError(Exception):
    """Stand-in for an SDK APIStatusError: carries a status_code and nothing else."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_poll_once_unreachable_raises_pending_not_failure(store):
    provider = _UnreachableProvider(_timeout_error(), on="poll")
    store.save_batch_id("k", "batch-123")
    with pytest.raises(BatchPendingError) as exc_info:
        _poll_once(provider, "batch-123", "k", store, label="Batch P1",
                   pass_name="p1", request_count=340)
    assert "unreachable" in exc_info.value.status
    assert exc_info.value.request_count == 340
    # The whole point: the batch id survives, so the next poll resumes it.
    assert store.get_batch_id("k") == "batch-123"


def test_poll_once_definite_http_error_still_fails(store):
    provider = _UnreachableProvider(_StatusError(401), on="poll")
    store.save_batch_id("k", "batch-123")
    with pytest.raises(_StatusError):
        _poll_once(provider, "batch-123", "k", store, label="Batch P1")


@pytest.mark.asyncio
async def test_collect_unreachable_raises_pending_and_keeps_batch_id(store):
    """A blip during collection must not throw away a finished, paid-for batch."""
    provider = _UnreachableProvider(_timeout_error(), on="collect", poll_sequence=["done"])
    with pytest.raises(BatchPendingError) as exc_info:
        await submit_and_poll(provider, [{"custom_id": "c1"}], store, "run9:p1",
                              label="Batch P1", pass_name="p1")
    assert "collect" in exc_info.value.status
    assert store.get_batch_id("run9:p1") == "batch-123"

    # Retry collects the same batch — no resubmission, no second charge.
    provider2 = _FakeProvider(["done"], collect_result=({"c1": "hello"}, {}, {}))
    results, _usage, _errors = await submit_and_poll(
        provider2, [{"custom_id": "c1"}], store, "run9:p1", label="Batch P1", pass_name="p1",
    )
    assert provider2.submit_calls == 0
    assert results == {"c1": "hello"}


# ---------------------------------------------------------------------------
# submit_and_poll
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_and_poll_pending_propagates_and_does_not_resubmit(store):
    provider = _FakeProvider(["running"])
    with pytest.raises(BatchPendingError):
        await submit_and_poll(provider, [{"custom_id": "c1"}], store, "run1:p1", label="Batch P1", pass_name="p1")
    assert provider.submit_calls == 1
    assert store.get_batch_id("run1:p1") == "batch-123"

    # Resume: a second call with the same batch_key must NOT re-submit.
    provider2 = _FakeProvider(["done"], collect_result=({"c1": "hello"}, {}, {}))
    # Simulate resumption by pointing a fresh provider at the same store (which
    # already holds the batch_id from the first call).
    results, usage, errors = await submit_and_poll(
        provider2, [{"custom_id": "c1"}], store, "run1:p1", label="Batch P1", pass_name="p1",
    )
    assert provider2.submit_calls == 0  # resumed, not resubmitted
    assert results == {"c1": "hello"}
    assert errors == {}
    assert store.get_batch_id("run1:p1") is None  # cleared on success


@pytest.mark.asyncio
async def test_submit_and_poll_returns_error_dict_excluding_failed_cids_from_results(store):
    collect_result = (
        {"c1": "ok text"},
        {"c1": {"input_tokens": 10}},
        {"c2": "errored: rate limited"},
    )
    provider = _FakeProvider(["done"], collect_result=collect_result)
    results, usage, errors = await submit_and_poll(
        provider, [{"custom_id": "c1"}, {"custom_id": "c2"}], store, "run2:p1", label="Batch P1",
    )
    assert "c2" not in results  # failed cid must never sentinel as "" success
    assert results == {"c1": "ok text"}
    assert errors == {"c2": "errored: rate limited"}


@pytest.mark.asyncio
async def test_submit_and_poll_failed_clears_kv(store):
    provider = _FakeProvider(["failed"])
    with pytest.raises(RuntimeError):
        await submit_and_poll(provider, [{"custom_id": "c1"}], store, "run3:p1", label="Batch P1")
    assert store.get_batch_id("run3:p1") is None


# ---------------------------------------------------------------------------
# Synthetic max_tokens=0 cache pre-warm (replaces spending a real paper's
# request on this — see _execute_pass1_with_groups in batch_runner.py). These
# cover the request-construction logic in isolation, at the same granularity
# as the rest of this file, without needing the full pipeline's paper/run/
# store scaffolding.
# ---------------------------------------------------------------------------

def _cached_two_user_message_list():
    """Mimics `caching.py`'s output for `cache_first="questions"`: [system,
    questions-block (cache_control on its last content block), paper-text
    block (plain string, uncached)]."""
    return [
        {"role": "system", "content": "You are an extractor."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer these questions...", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ],
        },
        {"role": "user", "content": "=== Paper text ===\n\nSome real paper content."},
    ]


def test_build_p1_request_max_tokens_zero_for_anthropic_prewarm():
    messages = _cached_two_user_message_list()
    request = build_p1_request(
        custom_id="run1-prewarm",
        provider="anthropic",
        model="claude-sonnet-5",
        messages=messages,
        temperature=0.0,
        model_params={"max_tokens": 0},
    )
    assert request["params"]["max_tokens"] == 0


def test_build_p1_request_max_tokens_zero_not_overridden_by_other_model_params():
    # {**cfg.model_params, "max_tokens": 0} must always win, regardless of what
    # a user's own model_params sets — mirrors the exact merge used in
    # _execute_pass1_with_groups's pre-warm branch.
    messages = _cached_two_user_message_list()
    cfg_model_params = {"max_tokens": 4096, "top_p": 0.9}
    request = build_p1_request(
        custom_id="run1-prewarm",
        provider="anthropic",
        model="claude-sonnet-5",
        messages=messages,
        temperature=0.0,
        model_params={**cfg_model_params, "max_tokens": 0},
    )
    assert request["params"]["max_tokens"] == 0
    assert request["params"]["top_p"] == 0.9


def test_prewarm_placeholder_keeps_cached_block_replaces_only_paper_text():
    """Exercises the exact transform _execute_pass1_with_groups applies to
    build a synthetic pre-warm request: deep-copy the first real request's
    messages, then overwrite only the last (uncached, paper-specific)
    message's content with a placeholder — the cache_control-marked block
    must survive untouched so the pre-warm hits the same cache entry a real
    request would."""
    first_messages = _cached_two_user_message_list()
    prewarm_messages = copy.deepcopy(first_messages)
    prewarm_messages[-1]["content"] = "warmup"

    # The cached (questions) block is untouched, cache_control intact.
    assert prewarm_messages[1] == first_messages[1]
    assert prewarm_messages[1]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Only the uncached (paper-text) block was replaced.
    assert prewarm_messages[-1]["content"] == "warmup"
    assert first_messages[-1]["content"] != "warmup"  # original untouched (deep copy, not alias)

    request = build_p1_request(
        custom_id="run1-prewarm",
        provider="anthropic",
        model="claude-sonnet-5",
        messages=prewarm_messages,
        temperature=0.0,
        model_params={"max_tokens": 0},
    )
    assert request["params"]["max_tokens"] == 0
    assert request["params"]["messages"][-1]["content"] == "warmup"


# ---------------------------------------------------------------------------
# AnthropicBatchProvider.prewarm
# ---------------------------------------------------------------------------
#
# Message Batches rejects max_tokens=0 outright ("max_tokens: must be greater
# than or equal to 1") — that's the actual bug these tests guard against
# regressing. prewarm() must call messages.create() directly, never
# beta.messages.batches.create(), so max_tokens=0 pre-warm requests are
# accepted rather than coming back as an errored batch item.

class _FakeMessagesClient:
    def __init__(self, raise_exc=None):
        self.create_calls = []
        self._raise_exc = raise_exc

    def create(self, **params):
        self.create_calls.append(params)
        if self._raise_exc is not None:
            raise self._raise_exc
        return object()


def _provider_with_fake_client(raise_exc=None):
    provider = AnthropicBatchProvider.__new__(AnthropicBatchProvider)
    fake_messages = _FakeMessagesClient(raise_exc=raise_exc)
    provider._client = type("FakeClient", (), {"messages": fake_messages})()
    return provider, fake_messages


def test_prewarm_calls_messages_create_directly_not_batches():
    provider, fake_messages = _provider_with_fake_client()
    params = {"model": "claude-sonnet-5", "max_tokens": 0, "messages": [{"role": "user", "content": "warmup"}]}

    provider.prewarm(params)

    assert fake_messages.create_calls == [params]


def test_prewarm_swallows_errors_instead_of_raising():
    provider, _ = _provider_with_fake_client(raise_exc=RuntimeError("boom"))
    params = {"model": "claude-sonnet-5", "max_tokens": 0, "messages": [{"role": "user", "content": "warmup"}]}

    provider.prewarm(params)  # must not raise — a failed pre-warm shouldn't abort the run


# ---------------------------------------------------------------------------
# _batch_item_cost
# ---------------------------------------------------------------------------
#
# Expected values are derived from litellm's own rate table rather than
# hard-coded, so a published price change does not break these tests. What is
# asserted is the shape of the calculation: each token class billed once, at
# its own rate, with the batch discount applied.
#
# The models named here are ones present in litellm's BUNDLED price map
# (model_prices_and_context_window_backup.json). litellm refreshes that map
# over the network at import time, and the newest model names exist only in
# the fetched copy — so a test naming one of those would pass online and fail
# offline (or under respx, which is active elsewhere in this suite).

def _rates(model, provider):
    import litellm

    info = litellm.get_model_info(model=model, custom_llm_provider=provider)
    return {k: Decimal(str(info[k])) for k in (
        "input_cost_per_token", "output_cost_per_token",
        "cache_creation_input_token_cost", "cache_read_input_token_cost",
    ) if info.get(k)} | {k: Decimal(str(info[k])) for k in (
        "input_cost_per_token_batches", "output_cost_per_token_batches",
    ) if info.get(k)}


def test_batch_cost_on_cache_hit_is_positive():
    """Regression: a fully-cached Anthropic item used to be priced as a refund.

    Anthropic reports input_tokens as the uncached part only, so anything that
    computes billable input as prompt_tokens - cache_read goes negative here.
    """
    cost = _batch_item_cost("anthropic", "claude-sonnet-4-5", {
        "input_tokens": 89_427, "output_tokens": 41_678,
        "cache_write_tokens": 0, "cache_read_tokens": 899_923,
    })

    assert cost > 0


def test_batch_cost_anthropic_bills_each_token_class_at_its_own_rate():
    r = _rates("claude-sonnet-4-5", "anthropic")
    usage = {
        "input_tokens": 89_427, "output_tokens": 48_413,
        "cache_write_tokens": 899_923, "cache_read_tokens": 12_000,
    }
    expected = Decimal("0.5") * (
        89_427 * r["input_cost_per_token"]
        + 899_923 * r["cache_creation_input_token_cost"]
        + 12_000 * r["cache_read_input_token_cost"]
        + 48_413 * r["output_cost_per_token"]
    )

    assert _batch_item_cost("anthropic", "claude-sonnet-4-5", usage) == expected


def test_batch_cost_openai_does_not_bill_cached_tokens_twice():
    """OpenAI's prompt_tokens already includes cached_tokens, unlike Anthropic's."""
    r = _rates("gpt-5.5", "openai")
    usage = {
        "input_tokens": 100_000, "output_tokens": 5_000,
        "cache_write_tokens": 0, "cache_read_tokens": 60_000,
    }
    expected = (
        40_000 * r["input_cost_per_token_batches"]          # published, already discounted
        + 60_000 * r["cache_read_input_token_cost"] * Decimal("0.5")
        + 5_000 * r["output_cost_per_token_batches"]
    )

    assert _batch_item_cost("openai", "gpt-5.5", usage) == expected


def test_batch_cost_unknown_model_returns_none_not_zero():
    """A missing price must leave the cost unset, not record a misleading $0."""
    assert _batch_item_cost("anthropic", "no-such-model-xyz", {
        "input_tokens": 1_000, "output_tokens": 10,
    }) is None


# ---------------------------------------------------------------------------
# Truncated Pass 1
#
# A Pass 1 that ran out of output budget stops mid-sentence, so the last question
# in the group has no answer block. Left in `p1_texts` it reaches Pass 2, which
# truthfully reports that answer as absent — and the run records a Pass-2 "key
# not found" error for what is a Pass-1 problem. See _demote_truncated_p1.
# ---------------------------------------------------------------------------

def test_truncated_pass1_becomes_an_error_not_an_answer():
    texts = {"cid1": "...built on top of embeddings to directly perform the ECPE"}
    usage = {"cid1": {"output_tokens": 4096, "finish_reason": "max_tokens"}}
    errors = {}

    _demote_truncated_p1(texts, usage, errors)

    assert "cid1" not in texts, "a truncated answer must not reach Pass 2"
    assert "cid1" in errors
    # The message has to name the real cause and the fix — the whole point is
    # that the old failure pointed at the wrong pass.
    assert "truncated" in errors["cid1"]
    assert "max output tokens" in errors["cid1"]


def test_openai_length_finish_reason_is_truncation_too():
    texts = {"cid1": "cut off"}
    usage = {"cid1": {"finish_reason": "length"}}
    errors = {}

    _demote_truncated_p1(texts, usage, errors)

    assert texts == {}
    assert "cid1" in errors


@pytest.mark.parametrize("reason", ["end_turn", "stop", "tool_use", "stop_sequence", None])
def test_a_model_that_finished_on_its_own_terms_is_left_alone(reason):
    texts = {"cid1": "a complete answer"}
    usage = {"cid1": {"finish_reason": reason}}
    errors = {}

    _demote_truncated_p1(texts, usage, errors)

    assert texts == {"cid1": "a complete answer"}
    assert errors == {}


def test_missing_usage_leaves_the_cell_alone():
    """A resumed run reloads Pass-1 texts from a dump with no usage alongside.
    Unknown is not truncated — never fail a cell on absent evidence."""
    texts = {"cid1": "text from a resume dump"}
    errors = {}

    _demote_truncated_p1(texts, {}, errors)

    assert texts == {"cid1": "text from a resume dump"}
    assert errors == {}


def test_demotion_is_idempotent():
    """It runs twice on the batch path — once before the resume dump is written,
    once on the way out with the online path."""
    texts = {"cid1": "cut off"}
    usage = {"cid1": {"finish_reason": "length"}}
    errors = {}

    _demote_truncated_p1(texts, usage, errors)
    first = dict(errors)
    _demote_truncated_p1(texts, usage, errors)

    assert errors == first


def test_only_the_truncated_cell_is_demoted():
    texts = {"good": "complete", "bad": "cut off"}
    usage = {
        "good": {"finish_reason": "end_turn"},
        "bad": {"finish_reason": "max_tokens"},
    }
    errors = {}

    _demote_truncated_p1(texts, usage, errors)

    assert texts == {"good": "complete"}
    assert list(errors) == ["bad"]


def test_anthropic_collect_records_the_stop_reason():
    """`collect` reports what the batch returned; the demotion decides what it
    means. Without the stop reason recorded here, truncation is invisible."""
    class _Usage:
        input_tokens = 1000
        output_tokens = 4096
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

    class _Block:
        text = "cut off mid-"

    class _Msg:
        content = [_Block()]
        usage = _Usage()
        stop_reason = "max_tokens"

    class _Result:
        type = "succeeded"
        message = _Msg()

    class _Item:
        custom_id = "cid1"
        result = _Result()

    class _Batches:
        def results(self, batch_id):
            return [_Item()]

    provider = AnthropicBatchProvider.__new__(AnthropicBatchProvider)
    client = type("C", (), {
        "beta": type("B", (), {"messages": type("M", (), {"batches": _Batches()})()})(),
    })()
    provider._client = client
    # Reads go through the retry-hardened copy of the client (see __init__).
    provider._read_client = client

    results, usage_by_cid, errors_by_cid = provider.collect("batch_x")

    assert usage_by_cid["cid1"]["finish_reason"] == "max_tokens"
    assert usage_by_cid["cid1"]["finish_reason"] in TRUNCATED_FINISH_REASONS


def test_default_pass1_budget_is_not_a_thinking_model_trap():
    """4096 was ~2000 usable tokens on a model whose omitted thinking still counts
    against max_tokens. The default must not be back in that range."""
    from seer_annotator.batch_runner import _DEFAULT_MAX_TOKENS_P1, _DEFAULT_MAX_TOKENS_P2

    assert _DEFAULT_MAX_TOKENS_P1 >= 16000
    assert _DEFAULT_MAX_TOKENS_P2 >= 4000


# ---------------------------------------------------------------------------
# Why one cell has no usable Pass-1 output
#
# _p1_drop_reason picks the most specific reason available and carries the
# evidence for it. The case it exists for: the online probe call fails, so the
# whole phase aborts and NO cell has a per-item error — every cell used to be
# written with the generic "no output (API error/timeout); see log" while the
# provider's actual message went only to the log.
# ---------------------------------------------------------------------------

def test_drop_reason_prefers_this_cells_own_error():
    detail, raw = _p1_drop_reason(
        "cid1",
        {"cid1": "RateLimitError: ... | status_code=429"},
        {},
        "AuthenticationError: ... | status_code=401",
    )

    assert detail.startswith("RateLimitError")
    assert raw["diagnostics"][0]["code"] == "pass1_provider_error"


def test_drop_reason_falls_back_to_the_phase_wide_abort():
    """A fatal Pass-1 error records no per-cell entry, but still explains them."""
    detail, _ = _p1_drop_reason("cid1", {}, {}, "AuthenticationError: ... | status_code=401")

    assert "AuthenticationError" in detail


def test_drop_reason_keeps_the_generic_sentence_as_a_last_resort():
    detail, _ = _p1_drop_reason("cid1", {}, {}, None)

    assert "no output (API error/timeout)" in detail


def test_drop_reason_carries_the_usage_that_proves_a_truncation():
    """A demoted Pass 1 has no text left — its usage is the only evidence."""
    usage = {"cid1": {"output_tokens": 4096, "max_tokens": 4096, "finish_reason": "max_tokens"}}

    _, raw = _p1_drop_reason("cid1", {"cid1": "pass1 truncated — ..."}, usage, None)

    assert raw["p1_usage"] == usage["cid1"]
    assert raw["diagnostics"][0]["code"] == "pass1_truncated"


def test_online_pass2_probe_failure_names_every_pending_cell():
    """Returning three empty dicts dropped those cells with no record at all.

    The caller's parse tail only iterates cids that HAVE Pass-2 text, and its
    error loop only iterates the error dict — a cid in neither was written
    nowhere and simply looked unanswered.
    """
    import asyncio

    run = SimpleNamespace(run_id=1, model_name="m", model_provider="openai")
    cfg = SimpleNamespace(
        format_model="fmt", format_model_provider="openai", format_model_params={},
        format_temperature=None, format_structured_output=False, batch_p2=False,
        request_timeout=None,
    )
    settings = SimpleNamespace(providers={})
    pending_cells = {
        "cid1": (SimpleNamespace(paper_id=1), [], 0),
        "cid2": (SimpleNamespace(paper_id=2), [], 0),
    }

    async def _boom(*a, **kw):
        raise RuntimeError("provider is on fire")

    with patch("seer_annotator.llm.complete", _boom):
        texts, usage, errors = asyncio.run(_execute_pass2(
            run, cfg, {"cid1": "p1", "cid2": "p1"}, pending_cells,
            store=None, settings=settings, dry_run=False,
        ))

    assert texts == {} and usage == {}
    assert set(errors) == {"cid1", "cid2"}
    assert "provider is on fire" in errors["cid1"]
