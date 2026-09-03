import json
from click.testing import CliRunner

from seer_annotator.benchmarking.evaluate import citation_match, evaluate
from seer_annotator.benchmarking.store import BenchmarkCase, BenchmarkStore, DatasetSpec, ModelConfig
from seer_annotator.cli import cli


def _make_store(tmp_path):
    store = BenchmarkStore(tmp_path / "b.sqlite")
    store.create_dataset(DatasetSpec("d", "source", {"runs": [[1, 1]]}))
    questions = [
        {"key": "bool", "question_type": "boolean", "allow_multiple": False},
        {"key": "cats", "question_type": "categorical", "allow_multiple": True},
        {"key": "float", "question_type": "float", "allow_multiple": False},
        {"key": "text", "question_type": "text", "allow_multiple": False},
    ]
    store.add_case(BenchmarkCase("d", "case", 1, 2, 3, "g", "Paper", "The source has enough words to verify a citation accurately.", "abstract", "reason", questions))
    for config in (ModelConfig("ref", "p", "ref"), ModelConfig("candidate", "p", "candidate")):
        store.add_model_config(config)
    return store


def _answer(value, key, **extra):
    return {"key": key, "value": value, "status": "ok", "cited_text": "", "comment": "", "confidence": 5, **extra}


def _save(store, model, answers, *, status="complete", diagnostics=None, raw_answers=None):
    store.ensure_executions("d", model)
    row = store.get_executions("d", model)[0]
    if row["status"] == "complete":
        with store.transaction() as con:
            con.execute("UPDATE executions SET status='pending' WHERE id=?", (row["id"],))
    store.claim_execution(row["id"], run_token="test")
    raw = json.dumps({"results": raw_answers if raw_answers is not None else answers})
    store.save_execution(row["id"], status=status, raw_response=raw, parsed_answers=answers,
                         diagnostics=diagnostics or {"native_json": True, "emitted_key_order": [a["key"] for a in answers]}, run_token="test")


def _make_store_two_cases(tmp_path):
    """Two-case variant of ``_make_store`` for tests that need one schema-valid
    group and one schema-failed group in the same report."""
    store = BenchmarkStore(tmp_path / "b2.sqlite")
    store.create_dataset(DatasetSpec("d", "source", {"runs": [[1, 1]]}))
    questions = [
        {"key": "bool", "question_type": "boolean", "allow_multiple": False},
        {"key": "cats", "question_type": "categorical", "allow_multiple": True},
        {"key": "float", "question_type": "float", "allow_multiple": False},
        {"key": "text", "question_type": "text", "allow_multiple": False},
    ]
    store.add_case(BenchmarkCase("d", "case1", 1, 2, 3, "g1", "Paper", "The source has enough words to verify a citation accurately.", "abstract", "reason", questions))
    store.add_case(BenchmarkCase("d", "case2", 1, 3, 4, "g2", "Paper", "The source has enough words to verify a citation accurately.", "abstract", "reason", questions))
    for config in (ModelConfig("ref", "p", "ref"), ModelConfig("candidate", "p", "candidate")):
        store.add_model_config(config)
    return store, questions


def _save_for_case(store, model, case_key, answers, *, status="complete", diagnostics=None, raw_answers=None):
    """Like ``_save`` but targets a specific case, for multi-case stores."""
    store.ensure_executions("d", model)
    row = next(r for r in store.get_executions("d", model) if r["case_key"] == case_key)
    if row["status"] == "complete":
        with store.transaction() as con:
            con.execute("UPDATE executions SET status='pending' WHERE id=?", (row["id"],))
    store.claim_execution(row["id"], run_token="test")
    raw = json.dumps({"results": raw_answers if raw_answers is not None else answers})
    store.save_execution(row["id"], status=status, raw_response=raw, parsed_answers=answers,
                         diagnostics=diagnostics or {"native_json": True, "emitted_key_order": [a["key"] for a in answers]}, run_token="test")


def test_citation_match_normalizes_and_pairs_multiple_quotes():
    assert citation_match([" Hello\nWorld ", "Second quote"], ["second   quote", "hello world"], .9)["match"]
    assert not citation_match(["one", "two"], ["one"], .9)["match"]
    assert citation_match([], "", .9)["match"]
    assert not citation_match("abcdef", "abcxyz", .9)["match"]


def test_evaluate_strict_types_multicat_comments_confidence_and_fuzzy(tmp_path):
    store = _make_store(tmp_path)
    ref = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text", cited_text="The source has enough words to verify a citation accurately.")]
    cand = [_answer(True, "bool"), _answer(["b", "a"], "cats"), _answer(1.5, "float"), _answer("quote", "text", cited_text="The source has enough words to verify a citation accurately!", comment="")]
    diag = {"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]}
    _save(store, "ref", ref, diagnostics=diag); _save(store, "candidate", cand, diagnostics=diag)
    report = evaluate(store, "d", "ref", ["candidate"], quote_threshold=.90)
    assert report["models"]["candidate"]["group_correct"] == 1
    assert report["models"]["candidate"]["citation_verification"]["answers_with_citations"] == 1
    cand[2] = _answer(1, "float")
    _save(store, "candidate", cand, diagnostics=diag)
    report = evaluate(store, "d", "ref", ["candidate"])
    assert report["models"]["candidate"]["group_correct"] == 0
    assert report["models"]["candidate"]["answer_correct"] == 3


def test_reference_invalid_excluded_and_candidate_missing_fails(tmp_path):
    store = _make_store(tmp_path)
    _save(store, "ref", [_answer(True, "bool")], diagnostics={"native_json": True, "emitted_key_order": ["bool"]})
    report = evaluate(store, "d", "ref", ["candidate"])
    assert report["excluded"]["reference_missing_or_invalid"] == 1
    # A valid reference and absent candidate count as a candidate failure.
    answers = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    _save(store, "ref", answers, diagnostics={"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]})
    report = evaluate(store, "d", "ref", ["candidate"])
    result = report["models"]["candidate"]
    assert result["candidate_missing"] == 1 and result["group_correct"] == 0


def test_evaluate_cli_writes_machine_reports_without_source_or_llm(tmp_path):
    store = _make_store(tmp_path)
    answers = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag = {"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]}
    _save(store, "ref", answers, diagnostics=diag)
    _save(store, "candidate", answers, diagnostics=diag)
    output_json, output_csv = tmp_path / "report.json", tmp_path / "report.csv"
    result = CliRunner().invoke(cli, ["benchmark", "evaluate", "--benchmark-db", str(tmp_path / "b.sqlite"), "--dataset", "d", "--reference", "ref", "--candidates", "candidate", "--json-out", str(output_json), "--csv-out", str(output_csv)])
    assert result.exit_code == 0, result.output
    assert json.loads(output_json.read_text())["models"]["candidate"]["group_correct"] == 1
    assert "candidate,case_key" in output_csv.read_text()


def test_missing_required_fields_and_invalid_status_make_reference_invalid(tmp_path):
    store = _make_store(tmp_path)
    complete = [_answer(True, "bool"), _answer(["a"], "cats"), _answer(1.5, "float"), _answer("x", "text")]
    required = ("key", "value", "cited_text", "comment", "confidence", "status")
    for field in required:
        broken = [dict(item) for item in complete]
        broken[0].pop(field)
        _save(store, "ref", broken, diagnostics={"native_json": True, "emitted_key_order": [a.get("key", "") for a in broken]})
        report = evaluate(store, "d", "ref", ["candidate"])
        assert report["excluded"].get("reference_missing_or_invalid") == 1, field
    broken = [dict(item) for item in complete]; broken[0]["status"] = "bogus"
    _save(store, "ref", broken, diagnostics={"native_json": True, "emitted_key_order": [a["key"] for a in broken]})
    assert evaluate(store, "d", "ref", ["candidate"])["excluded"]["reference_missing_or_invalid"] == 1


def test_diagnostic_records_missing_fields_and_invalid_status():
    from seer_annotator.annotate.parse import parse_structured_output_diagnostic
    diagnostic = parse_structured_output_diagnostic(json.dumps({"results": [{"key": "q", "value": True, "status": "bad"}]}), ["q"])
    assert diagnostic["missing_fields"]
    assert diagnostic["schema_violations"]


def test_no_direct_quote_sentinel_is_valid_and_not_verified(tmp_path):
    store = _make_store(tmp_path)
    reference = [_answer(True, "bool"), _answer(["a"], "cats"), _answer(1.5, "float"), _answer("x", "text", cited_text=None)]
    wire = [dict(item) for item in reference]
    wire[-1]["cited_text"] = "[NO DIRECT QUOTE]"
    diag = {"native_json": True, "emitted_key_order": [a["key"] for a in wire]}
    _save(store, "ref", reference, raw_answers=wire, diagnostics=diag)
    _save(store, "candidate", reference, raw_answers=wire, diagnostics=diag)
    report = evaluate(store, "d", "ref", ["candidate"])
    assert report["models"]["candidate"]["group_correct"] == 1
    assert report["models"]["candidate"]["citation_verification"]["answers_with_citations"] == 0


def test_components_verification_and_pending_coverage(tmp_path):
    store = _make_store(tmp_path)
    answers = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("x", "text", cited_text="The source has enough words to verify a citation accurately.")]
    diag = {"native_json": True, "emitted_key_order": [a["key"] for a in answers]}
    _save(store, "ref", answers, diagnostics=diag)
    store.ensure_executions("d", "candidate")
    execution = store.get_executions("d", "candidate")[0]
    report = evaluate(store, "d", "ref", ["candidate"])
    candidate = report["models"]["candidate"]
    assert candidate["coverage"]["candidate_groups"] == 0
    assert candidate["candidate_missing"] == 1
    assert candidate["components"]["citation"]["correct"] == 0
    assert candidate["citation_verification"]["answers_with_citations"] == 0


def test_cli_import_two_runs_and_evaluate_are_isolated(tmp_path, monkeypatch):
    """Exercise all three CLI stages with injected source/runner doubles."""
    import asyncio
    from seer_annotator.benchmarking.store import BenchmarkStore
    import seer_annotator.benchmarking.source as source_module
    import seer_annotator.benchmarking.runner as runner_module
    from seer_annotator.benchmarking.source import SourceRun

    pipeline = {
        "questions": [{"question_id": 1, "key": "q", "version": 1, "version_id": 1, "label": "Q", "question_type": "boolean"}],
        "papers": [{"paper_id": i, "title": f"Paper {i}", "abstract": f"Abstract {i}"} for i in range(1, 6)],
        "runs": [{"run_id": 7, "config": {"batching": "all"}}],
    }
    traces = [{"id": i, "paper_id": i, "group_id": f"7-{i}-0", "pass1_text": "reason", "pipeline": pipeline} for i in range(1, 6)]

    class FakeSource:
        def __init__(self, root):
            self.root = root
        def list_runs(self):
            return [SourceRun(7, "fake", "p1", "provider", {"batching": "all"}, usable_papers=5)]
        def traces_for_run(self, run_id): return traces
        def pipeline_for_run(self, run_id): return pipeline, "fake/pipeline.json"

    source_instances = []
    def source_factory(root):
        obj = FakeSource(root); source_instances.append(obj); return obj

    class FakeRunner:
        def __init__(self, store, **kwargs): self.store = store
        async def run(self, dataset, config, retry_errors=False):
            config_id = self.store.add_model_config(config); self.store.ensure_executions(dataset, config_id)
            for row in self.store.get_executions(dataset, config_id):
                if not self.store.claim_execution(row["id"], run_token="fake"):
                    continue
                answer = {"key": "q", "value": True, "status": "ok", "cited_text": "", "comment": "", "confidence": 5}
                self.store.save_execution(row["id"], status="complete", raw_response=json.dumps({"results": [answer]}), parsed_answers=[answer], diagnostics={"native_json": True, "emitted_key_order": ["q"]}, run_token="fake")
            return {"complete": 5, "error": 0, "skipped": 0}

    monkeypatch.setattr(source_module, "DjangoSource", source_factory)
    monkeypatch.setattr(runner_module, "BenchmarkRunner", FakeRunner)
    root = tmp_path / "source"; root.mkdir()
    settings = tmp_path / "settings.toml"; settings.write_text("")
    models = tmp_path / "models.toml"; models.write_text("[models.reference]\nprovider='p'\nmodel='r'\n[models.candidate]\nprovider='p'\nmodel='c'\n")
    db = tmp_path / "bench.sqlite"
    runner = CliRunner()
    imported = runner.invoke(cli, ["benchmark", "import-db", "--seer-root", str(root), "--benchmark-db", str(db), "--dataset", "smoke", "--run", "7:5", "--seed", "1"])
    assert imported.exit_code == 0, imported.output
    assert "Source PostgreSQL read-only mode verified" in imported.output
    for model in ("reference", "candidate"):
        result = runner.invoke(cli, ["benchmark", "run", "--benchmark-db", str(db), "--dataset", "smoke", "--settings", str(settings), "--models", str(models), "--model", model])
        assert result.exit_code == 0, result.output
    before_evaluate = len(source_instances)
    evaluated = runner.invoke(cli, ["benchmark", "evaluate", "--benchmark-db", str(db), "--dataset", "smoke", "--reference", "reference", "--candidates", "candidate"])
    assert evaluated.exit_code == 0, evaluated.output
    assert len(source_instances) == before_evaluate == 1


def test_components_and_breakdown_exclude_schema_failed_groups(tmp_path):
    """A group that fails the schema gate must not zero out every component.

    Regression for the bug where a failed group's answers were recorded as
    wrong on every component, making a schema/parsing problem look like a
    content problem repeated identically across every component and bucket.
    """
    store, questions = _make_store_two_cases(tmp_path)
    keys = [q["key"] for q in questions]
    good = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag_good = {"native_json": True, "emitted_key_order": keys}
    _save_for_case(store, "ref", "case1", good, diagnostics=diag_good)
    _save_for_case(store, "ref", "case2", good, diagnostics=diag_good)
    # case1: candidate is schema-valid and matches the reference exactly.
    _save_for_case(store, "candidate", "case1", good, diagnostics=diag_good)
    # case2: candidate's raw wire response is missing "comment" on one entry,
    # so it fails the schema gate; its stored parsed answers are otherwise
    # fine (a tolerant parser could have filled in a default), which is
    # exactly the situation that used to zero out every component.
    broken_raw = [dict(a) for a in good]
    del broken_raw[0]["comment"]
    _save_for_case(store, "candidate", "case2", good, diagnostics=diag_good, raw_answers=broken_raw)

    report = evaluate(store, "d", "ref", ["candidate"])
    candidate = report["models"]["candidate"]

    # The strict, group-gated numbers are untouched by this change.
    assert candidate["eligible_groups"] == 2
    assert candidate["group_correct"] == 1
    assert candidate["answer_total"] == 8

    # Components only count answers from the schema-valid group (case1).
    for component in ("status", "value", "confidence", "comment", "citation"):
        entry = candidate["components"][component]
        assert entry["total"] == 4
        assert entry["correct"] == 4
        assert entry["rate"] == 1.0
    # The excluded-count sibling records the answers left out of that
    # denominator so a caller never mistakes it for full coverage.
    assert candidate["components"]["excluded_answers"] == 4

    # Breakdown buckets get the same treatment, with a per-bucket exclusion count.
    for key in keys:
        bucket = candidate["breakdown"]["question_key"][key]
        assert bucket == {"total": 1, "correct": 1, "excluded": 1}


def test_schema_failure_reasons_attributes_cited_text_and_comment_violation(tmp_path):
    store, questions = _make_store_two_cases(tmp_path)
    keys = [q["key"] for q in questions]
    good = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag_good = {"native_json": True, "emitted_key_order": keys}
    _save_for_case(store, "ref", "case1", good, diagnostics=diag_good)
    _save_for_case(store, "ref", "case2", good, diagnostics=diag_good)
    _save_for_case(store, "candidate", "case1", good, diagnostics=diag_good)
    # case2's raw wire response has invalid (non-string) cited_text and
    # comment on the last entry -- a schema_violations entry with
    # fields=["cited_text", "comment"], matching the shape from the runner.
    bad_raw = [dict(a) for a in good]
    bad_raw[-1]["cited_text"] = 123
    bad_raw[-1]["comment"] = 456
    _save_for_case(store, "candidate", "case2", good, diagnostics=diag_good, raw_answers=bad_raw)

    report = evaluate(store, "d", "ref", ["candidate"])
    reasons = report["models"]["candidate"]["schema_failure_reasons"]

    assert reasons["cited_text"] == 1
    assert reasons["comment"] == 1
    # This failure is fully explained by schema_violations fields, so it must
    # not also be double-counted into the catch-all bucket.
    assert reasons.get("other", 0) == 0


def test_stale_stored_schema_violation_is_cleared_by_recompute(tmp_path):
    """A recompute from a present, re-parseable raw response is authoritative.

    Regression for the ``if parsed_diag.get(key):`` guard that could only add
    violations, never clear them -- so a database written by an older,
    stricter runner would report stale violations forever, even after the
    parser stopped considering them violations.
    """
    store = _make_store(tmp_path)
    good = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag_clean = {"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]}
    _save(store, "ref", good, diagnostics=diag_clean)
    # The stored diagnostics claim stale violations from an older run, but the
    # raw response saved alongside them (used by _save unless raw_answers is
    # given) is actually clean -- the recompute must win outright.
    stale_diag = {
        "native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"],
        "missing_keys": ["ghost"], "schema_violations": [{"ordinal": 0, "key": "bool", "fields": ["stale"]}],
    }
    _save(store, "candidate", good, diagnostics=stale_diag)

    report = evaluate(store, "d", "ref", ["candidate"])
    candidate = report["models"]["candidate"]

    assert candidate["group_correct"] == 1
    assert candidate["components"]["excluded_answers"] == 0
    assert candidate["components"]["status"]["total"] == 4
    assert candidate["schema_failure_reasons"] == {}


def _make_ghost_store(tmp_path, pass1_text, db_name="ghost.sqlite"):
    """Single-case store whose Pass-1 text is fully under the test's control,
    for exercising ghost-answer / false-absent detection."""
    store = BenchmarkStore(tmp_path / db_name)
    store.create_dataset(DatasetSpec("d", "source", {"runs": [[1, 1]]}))
    questions = [
        {"key": "bool", "question_type": "boolean", "allow_multiple": False},
        {"key": "cats", "question_type": "categorical", "allow_multiple": True},
        {"key": "float", "question_type": "float", "allow_multiple": False},
        {"key": "text", "question_type": "text", "allow_multiple": False},
    ]
    store.add_case(BenchmarkCase("d", "case", 1, 2, 3, "g", "Paper", "The source has enough words to verify a citation accurately.", "abstract", pass1_text, questions))
    for config in (ModelConfig("ref", "p", "ref"), ModelConfig("ghosty", "p", "ghosty"), ModelConfig("clean", "p", "clean")):
        store.add_model_config(config)
    return store


def test_ghost_answers_false_absent_and_clean_model(tmp_path):
    # "bool" and "text" have Pass-1 blocks; "cats" and "float" do not.
    pass1_text = (
        "--- ANSWER: bool ---\nQuotes:\n- \"yes it is\"\nAnswer: true\nConfidence: 15\n\n"
        "--- ANSWER: text ---\nQuotes:\n- \"a quote\"\nAnswer: some text\nConfidence: 10\n"
    )
    store = _make_ghost_store(tmp_path, pass1_text)
    ref = [_answer(True, "bool"), _answer(None, "cats", status="absent"), _answer(1.5, "float"), _answer("quote", "text")]
    _save(store, "ref", ref, diagnostics={"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]})

    # "ghosty" fabricates for both keys that have no Pass-1 block: a
    # non-null value for "cats" (worst case, reaches the DB) and a null-value
    # "unmappable" for "float" (still a fabrication, just without a value).
    # It also wrongly reports "absent" for "text", which DOES have a block --
    # that's false-absent, a different failure mode, and must not be counted
    # as a ghost.
    ghosty = [
        _answer(True, "bool"),
        _answer(["a"], "cats", status="ok"),
        _answer(None, "float", status="unmappable"),
        _answer(None, "text", status="absent"),
    ]
    _save(store, "ghosty", ghosty, diagnostics={"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]})

    # "clean" correctly reports "absent" for the two blockless keys and "ok"
    # for the two that have blocks.
    clean = [_answer(True, "bool"), _answer(None, "cats", status="absent"), _answer(None, "float", status="absent"), _answer("quote", "text")]
    _save(store, "clean", clean, diagnostics={"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]})

    report = evaluate(store, "d", "ref", ["ghosty", "clean"])
    ghosty_report = report["models"]["ghosty"]
    clean_report = report["models"]["clean"]

    assert ghosty_report["pass1_blocks_present"] == 2
    assert ghosty_report["pass1_blocks_absent"] == 2
    assert ghosty_report["ghost_answers"] == 2
    assert ghosty_report["ghost_answers_with_value"] == 1
    assert ghosty_report["false_absent"] == 1
    assert ghosty_report["ghost_rate"] == 1.0
    assert ghosty_report["false_absent_rate"] == 0.5

    # pass1_blocks_present/absent are a property of the cases, not the
    # candidate, so they must be identical across models on this dataset.
    assert clean_report["pass1_blocks_present"] == ghosty_report["pass1_blocks_present"]
    assert clean_report["pass1_blocks_absent"] == ghosty_report["pass1_blocks_absent"]
    assert clean_report["ghost_answers"] == 0
    assert clean_report["ghost_answers_with_value"] == 0
    assert clean_report["false_absent"] == 0
    assert clean_report["ghost_rate"] == 0.0
    assert clean_report["false_absent_rate"] == 0.0

    # Per-answer flags are retained on the answer rows so a ghost can be
    # traced back to its case/key without re-deriving anything.
    ghost_rows = [r for r in report["answer_rows"] if r["candidate"] == "ghosty" and r["ghost_answer"]]
    assert {r["question_key"] for r in ghost_rows} == {"cats", "float"}
    with_value_rows = [r for r in ghost_rows if r["ghost_answer_with_value"]]
    assert {r["question_key"] for r in with_value_rows} == {"cats"}
    false_absent_rows = [r for r in report["answer_rows"] if r["candidate"] == "ghosty" and r["false_absent"]]
    assert {r["question_key"] for r in false_absent_rows} == {"text"}

    # The CSV rows carry the same per-group counts.
    ghosty_row = next(r for r in report["rows"] if r["candidate"] == "ghosty")
    assert ghosty_row["ghost_answers"] == 2
    assert ghosty_row["ghost_answers_with_value"] == 1
    assert ghosty_row["false_absent"] == 1


def test_ghost_rate_denominator_guard_when_no_blocks_absent(tmp_path):
    # Every key has a Pass-1 block, so pass1_blocks_absent is 0 for every
    # model on this dataset -- ghost_rate must guard the division and read
    # 0.0 rather than raising or reporting None/NaN.
    pass1_text = (
        "--- ANSWER: bool ---\nAnswer: true\nConfidence: 10\n\n"
        "--- ANSWER: cats ---\nAnswer: a\nConfidence: 10\n\n"
        "--- ANSWER: float ---\nAnswer: 1.5\nConfidence: 10\n\n"
        "--- ANSWER: text ---\nAnswer: some text\nConfidence: 10\n"
    )
    store = _make_ghost_store(tmp_path, pass1_text, db_name="ghost_full.sqlite")
    ref = [_answer(True, "bool"), _answer(["a"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag = {"native_json": True, "emitted_key_order": ["bool", "cats", "float", "text"]}
    _save(store, "ref", ref, diagnostics=diag)
    _save(store, "clean", ref, diagnostics=diag)

    report = evaluate(store, "d", "ref", ["clean"])
    clean_report = report["models"]["clean"]
    assert clean_report["pass1_blocks_absent"] == 0
    assert clean_report["ghost_answers"] == 0
    assert clean_report["ghost_rate"] == 0.0
    assert clean_report["pass1_blocks_present"] == 4
    assert clean_report["false_absent_rate"] == 0.0


def test_valid_schema_reparses_emitted_order_and_answers_not_just_diag_lists(tmp_path):
    """A stale ``emitted_key_order``/``parsed_json`` must not hold back a
    recoverable raw response, and the comparison must use the recovered
    answers rather than the stale stored placeholders.

    Regression for a partial Change-3 recompute that only refreshed
    ``missing_keys``/``duplicate_keys``/``unexpected_keys``/``missing_fields``/
    ``schema_violations`` from the raw response but left ``emitted_key_order``
    and the answers used for comparison (``_answers(execution)``, reading the
    stored ``parsed_json``) untouched. That let a group whose raw response the
    current parser recovers perfectly still fail the gate on a stale stored
    ``emitted_key_order``, and -- had it passed -- would have compared against
    stale all-``None`` placeholder answers instead of the recovered values.
    """
    store = _make_store(tmp_path)
    keys = ["bool", "cats", "float", "text"]
    good = [_answer(True, "bool"), _answer(["a", "b"], "cats"), _answer(1.5, "float"), _answer("quote", "text")]
    diag_clean = {"native_json": True, "emitted_key_order": keys}
    _save(store, "ref", good, diagnostics=diag_clean)

    # Stored parsed_json/diagnostics are stale all-None placeholders -- as if
    # an older/broken parser had failed to recover this response at execution
    # time -- but the raw response saved alongside them is complete and
    # cleanly re-parseable by the current parser.
    stale_answers = [{"key": k, "value": None, "cited_text": "", "comment": "", "confidence": None, "status": "absent"} for k in keys]
    stale_diag = {"native_json": False, "fallback_used": True, "emitted_key_order": [], "missing_keys": list(keys)}
    _save(store, "candidate", stale_answers, diagnostics=stale_diag, raw_answers=good)

    report = evaluate(store, "d", "ref", ["candidate"])
    candidate = report["models"]["candidate"]

    # The group now passes the gate: recovered from raw, not held back by the
    # stale stored emitted_key_order/missing_keys.
    assert candidate["group_correct"] == 1
    assert candidate["components"]["excluded_answers"] == 0
    assert candidate["schema_failure_reasons"] == {}
    # And the comparison ran against the recovered values -- if it had used
    # the stale all-None placeholders, "value" would mismatch on every answer
    # instead of matching the reference on all four.
    assert candidate["components"]["value"]["correct"] == 4
    assert candidate["answer_correct"] == 4
