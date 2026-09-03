import pytest

from seer_annotator.benchmarking.store import BenchmarkCase, BenchmarkStore, DatasetSpec, ModelConfig


def test_store_migrates_and_freezes_dataset(tmp_path):
    path = tmp_path / "benchmark.sqlite"
    store = BenchmarkStore(path)
    assert store.get_dataset("missing") is None
    spec = DatasetSpec("small", "/source", {"runs": [[3, 5]]}, seed=9)
    assert store.create_dataset(spec) == store.create_dataset(spec)
    with pytest.raises(ValueError, match="different specification"):
        store.create_dataset(DatasetSpec("small", "/source", {"runs": [[3, 6]]}, seed=9))
    import sqlite3
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 2
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {
        "datasets", "source_runs", "cases", "model_configs", "executions", "answers"
    }
    con.close()


def test_case_and_model_are_idempotent_and_fingerprinted(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    store.create_dataset(DatasetSpec("small", "/source", {"runs": []}))
    case = BenchmarkCase("small", "1:2:1", 1, 2, 3, "1-2-0", "Paper", "abstract", "abstract", "reason", [{"key": "q"}])
    assert store.add_case(case) == store.add_case(case)
    cfg = ModelConfig("m", "openai", "model", params={"response_format": {"type": "json_object"}})
    assert store.add_model_config(cfg) == store.add_model_config(cfg)
    with pytest.raises(ValueError, match="different fingerprint"):
        store.add_model_config(ModelConfig("m", "openai", "other-model"))
    rows = store.get_cases("small")
    assert len(rows) == 1
    assert rows[0]["source_checksum"]
