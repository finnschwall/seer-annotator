import sqlite3

import pytest

from seer_annotator.benchmarking.source import (
    SourceRun,
    SourceSafetyError,
    SourceSnapshotter,
    postgres_read_only,
    stable_sample,
    source_environment,
    configure_postgres_read_only_options,
)
from seer_annotator.benchmarking.store import BenchmarkStore, DatasetSpec


def test_stable_sample_is_independent_and_repeatable():
    first = stable_sample([1, 2, 3, 4, 5], 10, 3, 99)
    assert first == stable_sample([5, 4, 3, 2, 1], 10, 3, 99)
    assert first != stable_sample([1, 2, 3, 4, 5], 11, 3, 99)
    assert stable_sample([1, 2], 10, 0, 99) == []


class FakeSource:
    def list_runs(self):
        return [SourceRun(7, "run", "p1", "provider", {"batching": "all"}, usable_papers=2)]

    def pipeline_for_run(self, run_id):
        return ({
            "questions": [
                {"question_id": 1, "key": "q1", "version": 1, "version_id": 11, "label": "Q1", "question_type": "text"},
                {"question_id": 2, "key": "q2", "version": 1, "version_id": 12, "label": "Q2", "question_type": "boolean"},
            ],
            "papers": [{"paper_id": 2, "title": "Two", "abstract": "A"}, {"paper_id": 3, "title": "Three", "abstract": "B"}],
            "runs": [{"run_id": 7, "config": {"batching": "all"}}],
        }, "/archive/pipeline.json")

    def traces_for_run(self, run_id):
        pipeline, _ = self.pipeline_for_run(run_id)
        return [
            {"id": 101, "paper_id": 2, "group_id": "7-2-0", "question_keys": [], "pass1_text": "reason two", "pipeline": pipeline},
            {"id": 102, "paper_id": 3, "group_id": "7-3-0", "question_keys": ["q1"], "pass1_text": "reason three", "pipeline": pipeline},
        ]


def test_snapshot_reconstructs_group_from_pipeline_not_trace_keys(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    snap = SourceSnapshotter(FakeSource(), store)
    result = snap.import_dataset(DatasetSpec("five", "/archive", {"runs": [[7, 2]]}), [(7, 2)])
    assert result["cases"] == 2
    rows = store.get_cases("five")
    assert [r["source_paper_id"] for r in rows] == [2, 3]
    assert all({q["key"] for q in __import__("json").loads(r["questions_json"])} == {"q1", "q2"} for r in rows)


def test_source_sqlite_is_read_only(tmp_path):
    source = tmp_path / "state.db"
    con = sqlite3.connect(source)
    con.execute("CREATE TABLE ocr_cache (paper_id INTEGER, markdown TEXT)")
    con.execute("INSERT INTO ocr_cache VALUES (1, 'text')")
    con.commit(); con.close()
    uri = f"file:{source}?mode=ro"
    ro = sqlite3.connect(uri, uri=True)
    assert ro.execute("SELECT markdown FROM ocr_cache WHERE paper_id=1").fetchone()[0] == "text"
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO ocr_cache VALUES (2, 'no')")
    ro.close()


def test_postgres_read_only_rejects_non_postgres_without_query():
    class Connection:
        vendor = "sqlite"
        queried = False
        def ensure_connection(self): self.queried = True
    connection = Connection()
    with pytest.raises(SourceSafetyError, match="PostgreSQL"):
        with postgres_read_only(connection):
            pass
    assert not connection.queried


def test_connection_options_are_configured_before_connection_open():
    class Connection:
        vendor = "postgresql"
        settings_dict = {"OPTIONS": {"connect_timeout": 3, "options": "-c statement_timeout=1000"}}
        opened = False
        def ensure_connection(self):
            assert self.settings_dict["OPTIONS"]["options"].endswith("-c default_transaction_read_only=on")
            self.opened = True
    connection = Connection()
    configure_postgres_read_only_options(connection)
    assert connection.settings_dict["OPTIONS"]["connect_timeout"] == 3
    assert "statement_timeout=1000" in connection.settings_dict["OPTIONS"]["options"]
    assert "default_transaction_read_only=on" in connection.settings_dict["OPTIONS"]["options"]
    connection.ensure_connection()
    assert connection.opened


def test_source_environment_disables_file_logging_and_restores_env(tmp_path, monkeypatch):
    (tmp_path / "settings.ini").write_text("[settings]\nPOSTGRES_DB=readonly\n")
    monkeypatch.setenv("LOG_FILE", "/must/not/open.log")
    with source_environment(tmp_path):
        assert __import__("os").environ["LOG_FILE"] == ""
        assert __import__("os").environ["POSTGRES_DB"] == "readonly"
    assert __import__("os").environ["LOG_FILE"] == "/must/not/open.log"


def test_case_conflict_is_rejected(tmp_path):
    from seer_annotator.benchmarking.store import BenchmarkCase
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    store.create_dataset(DatasetSpec("x", "/source", {}))
    case = BenchmarkCase("x", "same", 1, 2, 3, "g", "title", "text", "abstract", "p1", [{"key": "q"}])
    store.add_case(case)
    with pytest.raises(ValueError, match="different immutable"):
        store.add_case(BenchmarkCase("x", "same", 1, 2, 3, "g", "title", "CHANGED", "abstract", "p1", [{"key": "q"}]))


def test_full_text_override_never_falls_back_to_abstract(tmp_path):
    class FullTextSource(FakeSource):
        def traces_for_run(self, run_id):
            rows = super().traces_for_run(run_id)
            for row in rows:
                row["state_path"] = None
            return rows
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    snap = SourceSnapshotter(FullTextSource(), store)
    with pytest.raises(Exception, match="produced only 0 cases"):
        snap.import_dataset(DatasetSpec("full", "/archive", {}, case_kind="full_text"), [(7, 1)], allow_current_source=True)


def test_full_text_sampling_counts_only_papers_with_archived_text(tmp_path):
    class SelectiveFullText(FakeSource):
        def traces_for_run(self, run_id):
            rows = super().traces_for_run(run_id)
            rows[0]["state_path"] = "/archive/one.db"
            rows[1]["state_path"] = "/archive/two.db"
            return rows

        def full_text_from_state(self, state_path, paper_id):
            return "archived full text" if paper_id == 3 else None

    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    result = SourceSnapshotter(SelectiveFullText(), store).import_dataset(
        DatasetSpec("full", "/archive", {}, case_kind="full_text"), [(7, 1)]
    )
    assert result["cases"] == 1
    assert store.get_cases("full")[0]["source_paper_id"] == 3


def test_import_enforces_requested_validated_paper_count(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    with pytest.raises(Exception, match="only 2 importable papers; requested 3"):
        SourceSnapshotter(FakeSource(), store).import_dataset(
            DatasetSpec("too-many", "/archive", {}), [(7, 3)]
        )


def test_each_trace_uses_its_own_pipeline_archive(tmp_path):
    class PerTraceSource(FakeSource):
        def traces_for_run(self, run_id):
            first, _ = self.pipeline_for_run(run_id)
            second = {**first, "questions": [first["questions"][0]],
                      "runs": [{"run_id": 7, "config": {"batching": "all"}}]}
            return [
                {"id": 1, "paper_id": 2, "group_id": "7-2-0", "pass1_text": "a", "pipeline": first},
                {"id": 2, "paper_id": 3, "group_id": "7-3-0", "pass1_text": "b", "pipeline": second},
            ]
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    SourceSnapshotter(PerTraceSource(), store).import_dataset(DatasetSpec("per-trace", "/archive", {}), [(7, 2)])
    rows = store.get_cases("per-trace")
    assert [len(__import__("json").loads(row["questions_json"])) for row in rows] == [2, 1]
