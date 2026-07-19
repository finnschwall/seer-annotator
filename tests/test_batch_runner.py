"""Tests for the observable submit→poll batch redesign (batch_runner.py).

Covers: single-shot polling (BatchPendingError raised on non-terminal status,
no sleep loop), kv cleared on failure but NOT on pending, and the 3-tuple
collect()/submit_and_poll() return shape (results, usage_by_cid, errors_by_cid)
with failed cids excluded from `results` rather than sentineled as "".
"""

import copy

import pytest

from seer_annotator.batch_runner import (
    AnthropicBatchProvider,
    BatchPendingError,
    _poll_once,
    build_p1_request,
    submit_and_poll,
)
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
