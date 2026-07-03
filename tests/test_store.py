"""Tests for store idempotency and status logic."""

import pytest
from seer_annotator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def test_ocr_roundtrip(store):
    assert store.get_ocr(1) is None
    store.save_ocr(1, "hello markdown")
    assert store.get_ocr(1) == "hello markdown"


def test_ocr_none_stored(store):
    store.save_ocr(2, None)
    assert store.get_ocr(2) is None


def test_answer_lifecycle(store):
    assert store.get_status(1, 1, 1) is None
    store.upsert_pending(1, 1, 1)
    assert store.get_status(1, 1, 1) == "pending"

    store.save_answer(1, 1, 1, {"run": 1, "paper": 1, "question_version": 1, "value_text": "x"})
    assert store.get_status(1, 1, 1) == "done"
    assert store.should_skip_cell(1, 1, 1) is True  # done → don't recompute


def test_skip_posted(store):
    store.save_answer(1, 2, 3, {"run": 1})
    store.mark_posted(1, 2, [3])
    assert store.get_status(1, 2, 3) == "posted"
    assert store.should_skip_cell(1, 2, 3) is True


def test_idempotent_upsert_pending(store):
    store.upsert_pending(1, 1, 1)
    store.upsert_pending(1, 1, 1)  # no error
    assert store.get_status(1, 1, 1) == "pending"


def test_get_unposted(store):
    store.save_answer(1, 1, 10, {"run": 1, "paper": 1, "question_version": 10})
    store.save_answer(1, 1, 11, {"run": 1, "paper": 1, "question_version": 11})
    store.mark_posted(1, 1, [10])
    unposted = store.get_unposted(1, 1)
    assert len(unposted) == 1
    assert unposted[0]["question_version"] == 11


def test_stats(store):
    store.mark_skipped(1, 5, 99, "no_ocr")
    store.upsert_pending(1, 5, 100)
    s = store.stats()
    assert s["answers"]["skipped"] == 1
    assert s["answers"]["pending"] == 1


def test_reset_runs_clears_only_targeted_run(store):
    store.save_answer(1, 1, 1, {"run": 1})
    store.mark_posted(1, 1, [1])
    store.save_answer(2, 1, 1, {"run": 2})
    store.mark_posted(2, 1, [1])

    store.reset_runs([1])

    assert store.get_status(1, 1, 1) is None
    assert store.get_status(2, 1, 1) == "posted"


def test_reset_runs_empty_list_is_noop(store):
    store.save_answer(1, 1, 1, {"run": 1})
    store.mark_posted(1, 1, [1])

    store.reset_runs([])

    assert store.get_status(1, 1, 1) == "posted"


# ---------------------------------------------------------------------------
# Resolution (arbitration) lifecycle — mirrors the answers-table tests above,
# keyed by (arbiter_run_id, dispute_item_id) instead of (run_id, version_id).
# ---------------------------------------------------------------------------

def test_resolution_lifecycle(store):
    assert store.get_resolution_status(1, 501) is None
    store.upsert_pending_resolution(1, 501, 42, 14)
    assert store.get_resolution_status(1, 501) == "pending"

    store.save_pass1_resolution(1, 501, 42, 14, {"pass1_text": "..."})
    assert store.get_resolution_status(1, 501) == "pass1_done"

    store.save_resolution(1, 501, 42, 14, {"resolution_status": "ok", "dispute_item": 501})
    assert store.get_resolution_status(1, 501) == "done"
    assert store.should_skip_resolution_cell(1, 501) is True
    assert store.should_skip_resolution_cell_by_paper_version(1, 42, 14) is True


def test_resolution_skip_posted(store):
    store.save_resolution(1, 502, 42, 15, {"dispute_item": 502})
    store.mark_resolutions_posted(1, [502])
    assert store.get_resolution_status(1, 502) == "posted"
    assert store.should_skip_resolution_cell(1, 502) is True


def test_resolution_skip_by_paper_version_distinguishes_papers(store):
    """The same version_id disputed on two different papers must not collide."""
    store.save_resolution(1, 501, 42, 14, {})
    assert store.should_skip_resolution_cell_by_paper_version(1, 42, 14) is True
    assert store.should_skip_resolution_cell_by_paper_version(1, 99, 14) is False


def test_get_unposted_and_postable_resolutions(store):
    store.save_resolution(1, 501, 42, 14, {"dispute_item": 501})
    store.save_resolution(1, 502, 42, 15, {"dispute_item": 502})
    store.mark_resolutions_posted(1, [501])

    unposted = store.get_unposted_resolutions(1, 42)
    assert len(unposted) == 1
    assert unposted[0]["dispute_item"] == 502

    postable = store.get_postable_resolutions(1, 42)
    assert {p["dispute_item"] for p in postable} == {501, 502}


def test_resolution_stats(store):
    store.mark_resolution_skipped(1, 501, 42, 14, "no_ocr")
    store.upsert_pending_resolution(1, 502, 42, 15)
    stats = store.resolution_stats()
    assert stats["skipped"] == 1
    assert stats["pending"] == 1


def test_update_reformatted_resolution(store):
    store.save_resolution(1, 501, 42, 14, {"value_text": "old", "tokens_total": 5})
    store.update_reformatted_resolution(1, 501, {"value_text": "new", "tokens_total": 5})
    rows = store.get_reformattable_resolution_rows(1, 42)
    assert len(rows) == 1
    assert rows[0]["payload"]["value_text"] == "new"
    assert store.get_resolution_status(1, 501) == "done"
