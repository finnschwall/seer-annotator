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
