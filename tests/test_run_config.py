"""Tests for run-config resolution (effective_run_config / effective_arbiter_config).

Focus: `chunk_papers`, whose code default is conditional. Chunking gives a run
progress checkpoints, which is cheap online but expensive with the batch API —
there, one chunk means one provider batch that the job parks on and waits for,
so a small chunk size turns one wait into many.
"""

import pytest

from seer_annotator.config import (
    ArbiterRunConfig,
    ArbiterRunDefaults,
    CHUNK_PAPERS_DEFAULT,
    CHUNK_PAPERS_DEFAULT_BATCH,
    RunConfig,
    RunDefaults,
    effective_arbiter_config,
    effective_run_config,
)


def test_online_run_uses_the_small_default():
    cfg = effective_run_config(RunConfig(), RunDefaults())
    assert cfg.chunk_papers == CHUNK_PAPERS_DEFAULT


@pytest.mark.parametrize("field", ["batch_p1", "batch_p2"])
def test_batch_run_uses_the_batch_default(field):
    cfg = effective_run_config(RunConfig(**{field: True}), RunDefaults())
    assert cfg.chunk_papers == CHUNK_PAPERS_DEFAULT_BATCH


def test_explicit_chunk_papers_wins_over_the_batch_default():
    """Including a deliberately small one — the conditional default only fills a gap."""
    cfg = effective_run_config(RunConfig(batch_p1=True, chunk_papers=10), RunDefaults())
    assert cfg.chunk_papers == 10


def test_explicit_zero_is_kept():
    """0 means "all papers in one chunk" and must not be mistaken for unset."""
    cfg = effective_run_config(RunConfig(batch_p1=True, chunk_papers=0), RunDefaults())
    assert cfg.chunk_papers == 0


def test_settings_default_wins_over_the_batch_default():
    cfg = effective_run_config(RunConfig(batch_p1=True), RunDefaults(chunk_papers=25))
    assert cfg.chunk_papers == 25


def test_batch_default_is_below_the_provider_batch_ceiling():
    """A single Anthropic batch is capped at 100,000 requests / 256 MB, and one
    full-text request is ~200 KB — so the default must stay well under ~1,300."""
    assert CHUNK_PAPERS_DEFAULT < CHUNK_PAPERS_DEFAULT_BATCH < 1300


def test_arbitration_follows_the_same_rule():
    assert effective_arbiter_config(
        ArbiterRunConfig(), ArbiterRunDefaults()
    ).chunk_papers == CHUNK_PAPERS_DEFAULT
    assert effective_arbiter_config(
        ArbiterRunConfig(batch_p2=True), ArbiterRunDefaults()
    ).chunk_papers == CHUNK_PAPERS_DEFAULT_BATCH
