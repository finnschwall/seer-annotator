"""Tests for store idempotency and status logic."""

import pytest
from seer_annotator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def test_ocr_roundtrip(store):
    assert store.get_ocr(1, "") is None
    store.save_ocr(1, "hello markdown", "")
    assert store.get_ocr(1, "") == "hello markdown"


def test_ocr_none_stored(store):
    store.save_ocr(2, None, "")
    assert store.get_ocr(2, "") is None


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


# ---------------------------------------------------------------------------
# Delivery and retryability are separate facts
#
# `status` says whether the cell is recomputed on the next run; `posted_at` says
# whether SEER has the payload. A dropped cell (provider error, truncated pass)
# is both retryable and owed an error answer. While one column carried both,
# mark_failed cancelled the post, and the paper disappeared from the run with no
# error anywhere to explain it.
# ---------------------------------------------------------------------------

def test_failed_cell_is_still_unposted_work(store):
    store.save_answer(1, 1, 10, {"run": 1, "paper": 1, "question_version": 10})
    store.mark_failed(1, 1, 10, "pass1 truncated")

    unposted = store.get_unposted(1, 1)
    assert [p["question_version"] for p in unposted] == [10]


def test_posting_a_failed_cell_keeps_it_retryable(store):
    store.save_answer(1, 1, 10, {"run": 1, "paper": 1, "question_version": 10})
    store.mark_failed(1, 1, 10, "pass1 truncated")
    store.mark_posted(1, 1, [10])

    assert store.get_status(1, 1, 10) == "failed"
    assert store.should_skip_cell(1, 1, 10) is False   # next run tries it again
    assert store.get_unposted(1, 1) == []              # but it is not posted twice


def test_retried_cell_is_posted_again(store):
    """The retry's answer must reach SEER, or the error answer posted for the
    first attempt stays there as the run's final word."""
    store.save_answer(1, 1, 10, {"question_version": 10, "extraction_status": "error"})
    store.mark_failed(1, 1, 10, "boom")
    store.mark_posted(1, 1, [10])

    store.save_answer(1, 1, 10, {"question_version": 10, "extraction_status": "ok"})

    unposted = store.get_unposted(1, 1)
    assert [p["extraction_status"] for p in unposted] == ["ok"]


def test_get_unposted_ignores_half_processed_cells(store):
    store.save_pass1(1, 1, 10, {"question_version": 10, "pass1_text": "..."})
    assert store.get_unposted(1, 1) == []


def test_get_postable_includes_failed(store):
    store.save_answer(1, 1, 10, {"question_version": 10})
    store.mark_failed(1, 1, 10, "boom")
    store.save_answer(1, 1, 11, {"question_version": 11})
    store.mark_posted(1, 1, [11])

    postable = store.get_postable(1, 1)
    assert {p["question_version"] for p in postable} == {10, 11}


def test_store_file_written_before_posted_at_is_upgraded(tmp_path):
    """A job store outlives the code that wrote it — a parked batch job is resumed
    days later from the file on disk. Rows already posted must not be posted again."""
    import sqlite3

    path = str(tmp_path / "legacy.db")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE answers (
            run_id INTEGER NOT NULL, paper_id INTEGER NOT NULL, version_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', payload_json TEXT, batch_group_id TEXT,
            error TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, paper_id, version_id)
        );
        INSERT INTO answers (run_id, paper_id, version_id, status, payload_json, updated_at)
        VALUES (1, 1, 10, 'posted', '{"question_version": 10}', 'then'),
               (1, 1, 11, 'done',   '{"question_version": 11}', 'then'),
               (1, 1, 12, 'failed', '{"question_version": 12}', 'then');
    """)
    con.commit()
    con.close()

    store = Store(path)

    # 11 was never delivered; 12 is the error answer the old code dropped.
    assert {p["question_version"] for p in store.get_unposted(1, 1)} == {11, 12}


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


def test_failed_resolution_is_posted_and_stays_retryable(store):
    """The resolutions twin of test_posting_a_failed_cell_keeps_it_retryable —
    a dropped dispute item owes SEER an error resolution just as a dropped
    annotation cell owes it an error answer."""
    store.save_resolution(1, 503, 42, 16, {"dispute_item": 503})
    store.mark_resolution_failed(1, 503, "provider error")

    assert [r["dispute_item"] for r in store.get_unposted_resolutions(1, 42)] == [503]

    store.mark_resolutions_posted(1, [503])

    assert store.get_resolution_status(1, 503) == "failed"
    assert store.should_skip_resolution_cell(1, 503) is False
    assert store.get_unposted_resolutions(1, 42) == []
    assert [r["dispute_item"] for r in store.get_postable_resolutions(1, 42)] == [503]


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


# ---------------------------------------------------------------------------
# Resume-safe progress counters (finished_cells / finished_resolutions)
#
# These exist because a batch run parks on BatchPendingError and re-enters
# run_pipeline from the top: a counter starting at 0 reports only the last
# invocation's work, and SEER downgrades a "succeeded" heartbeat short of
# cells_total to "failed". Returned per cell rather than as a total because a
# resume re-processes every cell that is not done/posted. See Store.finished_cells.
# ---------------------------------------------------------------------------

PAPERS = [1, 2]
VERSIONS = [10, 11]


def _ok(value="x"):
    return {"extraction_status": "ok", "value_text": value}


def _err():
    return {"extraction_status": "error", "extraction_detail": "boom"}


def _invalid():
    return {"extraction_status": "invalid", "extraction_detail": "unmappable value"}


def test_finished_cells_empty(store):
    assert store.finished_cells(1, PAPERS, VERSIONS) == {}


def test_finished_cells_reports_each_cell_once(store):
    store.save_answer(1, 1, 10, _ok())
    store.save_answer(1, 1, 11, _ok())
    store.save_answer(1, 2, 10, _err())
    assert store.finished_cells(1, PAPERS, VERSIONS) == {
        (1, 10): False, (1, 11): False, (2, 10): True,
    }


def test_finished_cells_counts_invalid_as_error(store):
    """`invalid` is an answer nobody can use — it belongs in cells_error, and the
    live counter (mapping.payload_is_error) must agree with this."""
    store.save_answer(1, 1, 10, _invalid())
    assert store.finished_cells(1, PAPERS, VERSIONS) == {(1, 10): True}


def test_finished_cells_survives_posted_and_failed(store):
    """mark_posted/mark_failed change the row's status but keep its payload: the
    cell was still counted once when its payload was saved."""
    store.save_answer(1, 1, 10, _ok())
    store.mark_posted(1, 1, [10])
    store.save_answer(1, 1, 11, _err())
    store.mark_failed(1, 1, 11, "boom")
    assert store.finished_cells(1, PAPERS, VERSIONS) == {(1, 10): False, (1, 11): True}


def test_finished_cells_ignores_pending_and_pass1(store):
    """A pending row has no payload and a pass1_done row is only half processed —
    neither has gone through _count_payload yet."""
    store.upsert_pending(1, 1, 10)
    store.save_pass1(1, 1, 11, {"extraction_status": "ok"})
    assert store.finished_cells(1, PAPERS, VERSIONS) == {}


def test_finished_cells_is_scoped(store):
    """Other runs, and papers/versions outside the caller's scope, never appear —
    the seed must not be able to exceed that scope's cells_total."""
    store.save_answer(1, 1, 10, _ok())
    store.save_answer(1, 99, 10, _ok())   # paper out of scope
    store.save_answer(1, 1, 99, _ok())    # question version out of scope
    store.save_answer(2, 1, 10, _ok())    # different run
    assert store.finished_cells(1, PAPERS, VERSIONS) == {(1, 10): False}


def test_finished_resolutions(store):
    store.save_resolution(1, 501, 42, 14, {"resolution_status": "ok"})
    store.save_resolution(1, 502, 42, 15, {"resolution_status": "invalid"})
    store.mark_resolutions_posted(1, [501])
    store.save_pass1_resolution(1, 503, 42, 16, {"resolution_status": "ok"})
    store.upsert_pending_resolution(1, 504, 42, 17)
    store.save_resolution(2, 505, 42, 18, {"resolution_status": "ok"})  # other arbiter run

    assert store.finished_resolutions(1, [501, 502, 503, 504]) == {501: False, 502: True}
    assert store.finished_resolutions(1, [501]) == {501: False}


def test_finished_cells_tolerates_unreadable_payload(store):
    """A row whose payload can't be parsed still counts as done — dropping it would
    make a finished run look unfinished, which is the failure this seed prevents."""
    store.save_answer(1, 1, 10, _ok())
    with store._tx() as con:
        con.execute(
            "UPDATE answers SET payload_json='{not json' WHERE run_id=1 AND paper_id=1 AND version_id=10"
        )
    assert store.finished_cells(1, PAPERS, VERSIONS) == {(1, 10): False}
