"""Offline comparison of stored Phase-2 benchmark executions.

This module deliberately has no model or source-database dependency.  It reads
only the immutable cases and execution rows in :class:`BenchmarkStore`; this
makes it safe to run repeatedly, long after the expensive model calls finish.
"""

from __future__ import annotations

import csv
import difflib
import json
import itertools
import re
import unicodedata
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..annotate.verify import verify_citations
from ..annotate.parse import parse_structured_output_diagnostic, pass1_block_present
from .store import BenchmarkStore


def _norm_quote(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def _quotes(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [_norm_quote(v) for v in value if v is not None and _norm_quote(v)]
    text = _norm_quote(value)
    return [text] if text else []


def citation_match(reference: Any, candidate: Any, threshold: float = .90) -> dict[str, Any]:
    """Compare quote lists using a maximum one-to-one fuzzy matching.

    The equal-cardinality check is intentional: dropping a supporting quote is
    a formatting error even when the remaining quote is an excellent match.
    """
    refs, tests = _quotes(reference), _quotes(candidate)
    if not refs and not tests:
        return {"match": True, "reference_count": 0, "candidate_count": 0, "similarity": None}
    if len(refs) != len(tests):
        return {"match": False, "reference_count": len(refs), "candidate_count": len(tests), "similarity": None}
    scores = [[difflib.SequenceMatcher(None, r, t).ratio() for t in tests] for r in refs]
    # Maximize total similarity first, then apply the per-pair threshold.  The
    # DP is exponential only in quote count (normally 1-3), and is deterministic
    # on ties because lower candidate indexes are visited first.
    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> tuple[float, tuple[int, ...]]:
        if i == len(refs):
            return 0.0, ()
        winner: tuple[float, tuple[int, ...]] | None = None
        for j in range(len(tests)):
            if used & (1 << j):
                continue
            tail_score, tail_assignment = best(i + 1, used | (1 << j))
            option = (scores[i][j] + tail_score, (j,) + tail_assignment)
            if winner is None or option[0] > winner[0] + 1e-12:
                winner = option
        return winner or (-float("inf"), ())
    total, assignment = best(0, 0)
    selected = [scores[i][assignment[i]] for i in range(len(refs))] if assignment else []
    ok = len(selected) == len(refs) and all(score >= threshold for score in selected)
    average = sum(selected) / len(selected) if selected else None
    return {"match": ok, "reference_count": len(refs), "candidate_count": len(tests), "similarity": average}


def _value_equal(reference: Any, candidate: Any, question: Mapping[str, Any]) -> bool:
    """Type-aware comparison, with unordered values for categorical-multi."""
    qtype = str(question.get("question_type", "text"))
    multiple = bool(question.get("allow_multiple"))
    if multiple or (qtype == "categorical" and isinstance(reference, list)):
        if not isinstance(reference, list) or not isinstance(candidate, list):
            return False
        # Values are option strings and categorical-multi is semantically a
        # set: ordering and accidental duplicate emission do not matter.
        try:
            return set(reference) == set(candidate)
        except TypeError:
            return sorted(reference, key=lambda x: json.dumps(x, sort_keys=True)) == sorted(candidate, key=lambda x: json.dumps(x, sort_keys=True))
    if reference is None or candidate is None:
        return reference is None and candidate is None
    if type(reference) is not type(candidate):
        return False
    return reference == candidate


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else (value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _diag(execution: Mapping[str, Any]) -> dict[str, Any]:
    return _loads(execution.get("diagnostics_json"), {}) or {}


def _answers(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = _loads(execution.get("parsed_json"), [])
    return parsed if isinstance(parsed, list) else []


def _valid_schema(execution: Mapping[str, Any], expected: list[str]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Return ``(is_valid, diagnostics_used, answers_used)``.

    The stored ``parsed_json``/``diagnostics_json`` are only a cache of
    whatever parser produced them at execution time. ``raw_response`` is the
    durable record of what the model actually said, so whenever it is
    present and re-parseable, evaluation re-derives *everything* the gate and
    the answer comparison consume -- diagnostics AND answers, including
    ``emitted_key_order`` -- from that one reparse, wholesale, rather than
    selectively refreshing a subset of diagnostic keys while leaving
    ``emitted_key_order`` and the answers themselves stale. That's what makes
    re-evaluating an old database actually pick up parser improvements (a
    repaired truncated response, relaxed null handling, etc.) instead of
    silently gating on, or comparing against, last time's parse. The stored
    diagnostics/answers are used only when there is no usable raw response to
    recompute from.
    """
    if execution.get("status") != "complete":
        return False, {}, []
    raw = execution.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        diag = parse_structured_output_diagnostic(raw, expected, annotate_mode=True)
        answers = diag.get("answers") or []
    else:
        diag = _diag(execution)
        answers = _answers(execution)
    if (diag.get("missing_keys") or diag.get("duplicate_keys") or diag.get("unexpected_keys") or
            diag.get("missing_fields") or diag.get("schema_violations")):
        return False, diag, answers
    emitted = diag.get("emitted_key_order")
    if emitted != expected or [str(a.get("key", "")) for a in answers] != expected:
        return False, diag, answers
    required = ("key", "value", "cited_text", "comment", "confidence", "status")
    for answer in answers:
        if any(field not in answer for field in required):
            return False, diag, answers
        if answer.get("status") not in {"ok", "absent", "unmappable"}:
            return False, diag, answers
        if not isinstance(answer.get("key"), str) or not isinstance(answer.get("comment"), str):
            return False, diag, answers
        cited = answer.get("cited_text")
        # The parser canonicalizes the valid wire sentinel ``[NO DIRECT
        # QUOTE]`` to None. The diagnostics above already verify that the
        # emitted cited_text field existed and had a valid wire type.
        if cited is not None and not (isinstance(cited, str) or (isinstance(cited, list) and all(isinstance(v, str) for v in cited))):
            return False, diag, answers
        confidence = answer.get("confidence")
        if confidence is not None and (not isinstance(confidence, int) or isinstance(confidence, bool)):
            return False, diag, answers
    return True, diag, answers


def _accumulate_schema_failure_reasons(counter: dict[str, int], diag: Mapping[str, Any]) -> None:
    """Attribute one failed-gate group to the diagnostics that explain it.

    Mirrors the per-answer ``fields`` breakdown for ``schema_violations`` and
    plain counts for the other diagnostic lists.  When none of them are
    populated -- e.g. a non-``complete`` execution status, an emitted/stored
    key-order mismatch, or a stored-answer field the raw-response reparse
    above didn't itself flag -- the failure is real but not explained by any
    tracked diagnostic, so it falls into ``other``.
    """
    explained = False
    for violation in diag.get("schema_violations") or []:
        for field in violation.get("fields") or ():
            counter[str(field)] += 1
            explained = True
    for key in ("missing_keys", "duplicate_keys", "unexpected_keys", "missing_fields"):
        count = len(diag.get(key) or [])
        if count:
            counter[key] += count
            explained = True
    if not explained:
        counter["other"] += 1


_COMPONENTS = ("status", "value", "confidence", "comment", "citation")


def _answer_comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any], question: Mapping[str, Any], threshold: float, source: str) -> dict[str, Any]:
    cite = citation_match(reference.get("cited_text"), candidate.get("cited_text"), threshold)
    result = {
        "status": reference.get("status", "ok") == candidate.get("status", "ok"),
        "value": _value_equal(reference.get("value"), candidate.get("value"), question),
        "confidence": reference.get("confidence") == candidate.get("confidence"),
        "comment": (reference.get("comment") or "") == (candidate.get("comment") or ""),
        "citation": cite,
    }
    result["match"] = all((result[k] for k in ("status", "value", "confidence", "comment"))) and bool(cite["match"])
    # Verification is diagnostic only; it never changes the model-vs-model score.
    try:
        result["candidate_citation_verified"] = verify_citations(candidate.get("cited_text"), source)
    except Exception as exc:  # malformed stored data must not make evaluation abort
        result["candidate_citation_verified"] = [{"ok": False, "note": str(exc)}]
    return result


def evaluate(
    store: BenchmarkStore,
    dataset: str,
    reference: str,
    candidates: Iterable[str] | None = None,
    *,
    quote_threshold: float = .90,
) -> dict[str, Any]:
    """Evaluate candidates against a stored reference and return JSON-safe data."""
    if not 0 <= quote_threshold <= 1:
        raise ValueError("quote threshold must be between 0 and 1")
    ref_row = store.get_model_config(reference)
    if not ref_row:
        raise KeyError(f"unknown reference model config {reference!r}")
    configs = store.get_model_configs()
    names = [str(c) for c in (candidates or [])]
    if not names:
        names = [str(c["name"]) for c in configs if c["name"] != ref_row["name"]]
    if "all" in names:
        names = [str(c["name"]) for c in configs if c["name"] != ref_row["name"]]
    names = list(dict.fromkeys(names))
    # Ensure candidates exist before producing a partially valid report.
    missing = [n for n in names if not store.get_model_config(n)]
    if missing:
        raise KeyError(f"unknown candidate model config(s): {', '.join(missing)}")
    cases = store.get_cases(dataset)
    ref_exec = {r["case_key"]: r for r in store.get_executions(dataset, ref_row["name"])}
    candidate_exec = {name: {r["case_key"]: r for r in store.get_executions(dataset, name)} for name in names}
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    excluded = defaultdict(int)
    for case in cases:
        questions = _loads(case.get("questions_json"), [])
        keys = [str(q.get("key", "")) for q in questions]
        ref = ref_exec.get(case["case_key"])
        ref_valid, _ref_diag, ref_answers = _valid_schema(ref, keys) if ref else (False, {}, [])
        if not ref or not ref_valid:
            excluded["reference_missing_or_invalid"] += 1
            continue
        for name in names:
            cand = candidate_exec[name].get(case["case_key"])
            valid_cand, cand_gate_diag, cand_answers = _valid_schema(cand, keys) if cand else (False, {}, [])
            by_key = {str(a.get("key")): a for a in cand_answers}
            # Ghost/false-absent detection deliberately reads the answer AS
            # ``stored_by_key`` is what the pipeline actually persisted at
            # execution time (``parsed_json`` / the ``answers`` table). It is
            # kept only to report how much an older parser lost: whenever it
            # could not extract a key it wrote status="absent", so a model that
            # emitted valid-but-awkward JSON looks in stored data as though it
            # claimed "absent" when it never did. The ghost/false-absent
            # metrics below therefore read the model's claim from ``by_key``
            # (re-derived from ``raw_response`` with the current parser, the
            # same source the accuracy metrics use) — this report exists to
            # rank models, and charging a model for a parser limitation ranks
            # them wrongly. ``parse_recovered`` is the gap between the two.
            stored_by_key = {str(a.get("key")): a for a in (_answers(cand) if cand else [])}
            answer_results = []
            for index, question in enumerate(questions):
                ra = ref_answers[index] if index < len(ref_answers) else {}
                ca = by_key.get(str(question.get("key")), {})
                answer_results.append(_answer_comparison(ra, ca, question, quote_threshold, case.get("source_text", "")) if valid_cand else {"match": False, "status": False, "value": False, "confidence": False, "comment": False, "citation": {"match": False}})
            group_match = valid_cand and all(a["match"] for a in answer_results)
            diag = _diag(cand) if cand else {}
            row = {"candidate": name, "case_key": case["case_key"], "source_run": case["source_run_id"], "group_size": len(questions), "group_match": bool(group_match), "candidate_status": cand.get("status") if cand else "missing", "source_text_kind": case.get("text_kind", ""),
                   # Reference-free ghost/false-absent counts for this group,
                   # filled in below alongside ``report["ghost_answers"]`` etc.
                   "pass1_blocks_present": 0, "pass1_blocks_absent": 0, "ghost_answers": 0, "ghost_answers_with_value": 0, "false_absent": 0}
            rows.append(row)
            reports.setdefault(name, {"candidate": name, "eligible_groups": 0, "group_correct": 0, "candidate_missing": 0, "candidate_incomplete": 0, "candidate_errors": 0, "answer_total": 0, "answer_correct": 0, "coverage": {"reference_groups": 0, "candidate_groups": 0}, "cost": 0.0, "tokens": {"input": 0, "output": 0, "cached": 0}, "latency_ms": {"total": 0, "count": 0, "average": None}, "citation_verification": {"answers_with_citations": 0, "all_verified": 0, "failures": 0},
                # ``components`` and ``breakdown`` count only answers from
                # schema-valid candidate groups (see ``valid_cand`` below), so
                # their rates answer "when this model's output is usable,
                # what does it get wrong" instead of replicating the group
                # gate onto every component/bucket. ``excluded_answers`` /
                # each bucket's ``excluded`` record how many answers were
                # left out of that denominator so the missing coverage is
                # never mistaken for 100%.
                "components": {**{component: {"total": 0, "correct": 0} for component in _COMPONENTS}, "excluded_answers": 0},
                # Reference-free ghost/false-absent bookkeeping (see
                # ``pass1_block_present``): computed from this candidate's own
                # stored answers plus the frozen ``pass1_text``, independent
                # of the reference and of the schema gate below. Counted over
                # every eligible-group answer, exactly the population
                # ``components``/``breakdown`` count.
                "pass1_blocks_present": 0, "pass1_blocks_absent": 0,
                "ghost_answers": 0, "ghost_answers_with_value": 0,
                "false_absent": 0, "parse_recovered": 0,
                "schema_failure_reasons": defaultdict(int),
                "diagnostics": {"native_json": 0, "schema_compliant": 0, "fallback_used": 0, "repair_used": 0, "missing_keys": 0, "duplicate_keys": 0, "unexpected_keys": 0, "missing_fields": 0, "schema_violations": 0}, "breakdown": {"source_run": {}, "question_key": {}, "question_type": {}, "group_size": {}}})
            report = reports[name]
            report["eligible_groups"] += 1
            report["coverage"]["reference_groups"] += 1
            report["coverage"]["candidate_groups"] += int(cand is not None and cand.get("status") == "complete")
            report["group_correct"] += int(group_match)
            if not cand or cand.get("status") in {"pending", "running"}:
                report["candidate_missing"] += 1
                if cand:
                    report["candidate_incomplete"] += 1
            elif cand.get("status") == "error": report["candidate_errors"] += 1
            if not valid_cand:
                _accumulate_schema_failure_reasons(report["schema_failure_reasons"], cand_gate_diag)
            for key in ("native_json", "fallback_used", "repair_used"):
                report["diagnostics"][key] += int(bool(diag.get(key)))
            report["diagnostics"]["schema_compliant"] += int(valid_cand)
            for key in ("missing_keys", "duplicate_keys", "unexpected_keys"):
                report["diagnostics"][key] += len(diag.get(key) or [])
            report["diagnostics"]["missing_fields"] += len(diag.get("missing_fields") or [])
            report["diagnostics"]["schema_violations"] += len(diag.get("schema_violations") or [])
            for index, question in enumerate(questions):
                ar = answer_results[index]
                report["answer_total"] += 1; report["answer_correct"] += int(ar["match"])
                if valid_cand:
                    for component in _COMPONENTS:
                        report["components"][component]["total"] += 1
                        component_value = ar.get(component)
                        if component == "citation" and isinstance(component_value, Mapping):
                            component_value = component_value.get("match")
                        report["components"][component]["correct"] += int(bool(component_value))
                else:
                    report["components"]["excluded_answers"] += 1
                verification = ar.get("candidate_citation_verified") or []
                cited = bool(_quotes(by_key.get(str(question.get("key")), {}).get("cited_text"))) if valid_cand else False
                if cited:
                    report["citation_verification"]["answers_with_citations"] += 1
                    if verification and all(bool(item.get("ok")) for item in verification):
                        report["citation_verification"]["all_verified"] += 1
                    else:
                        report["citation_verification"]["failures"] += 1
                # Ghost/false-absent bookkeeping: reference-free, so it reads
                # straight from this candidate's own raw answer (``by_key``,
                # not gated by ``valid_cand``) and the frozen ``pass1_text``.
                key_str = str(question.get("key"))
                raw_answer = by_key.get(key_str, {})
                block_present = pass1_block_present(case.get("pass1_text", "") or "", key_str)
                raw_status = raw_answer.get("status")
                if stored_by_key.get(key_str, {}).get("status") == "absent" and raw_status != "absent":
                    report["parse_recovered"] += 1
                is_ghost = (not block_present) and raw_status in ("ok", "unmappable")
                is_ghost_with_value = is_ghost and raw_answer.get("value") is not None
                is_false_absent = block_present and raw_status == "absent"
                report["pass1_blocks_present" if block_present else "pass1_blocks_absent"] += 1
                report["ghost_answers"] += int(is_ghost)
                report["ghost_answers_with_value"] += int(is_ghost_with_value)
                report["false_absent"] += int(is_false_absent)
                row["pass1_blocks_present" if block_present else "pass1_blocks_absent"] += 1
                row["ghost_answers"] += int(is_ghost)
                row["ghost_answers_with_value"] += int(is_ghost_with_value)
                row["false_absent"] += int(is_false_absent)
                answer_rows.append({"candidate": name, "case_key": case["case_key"], "source_run": case["source_run_id"], "question_key": question.get("key"), "question_type": question.get("question_type", ""), "match": bool(ar["match"]), "status_match": bool(ar.get("status")), "value_match": bool(ar.get("value")), "confidence_match": bool(ar.get("confidence")), "comment_match": bool(ar.get("comment")), "citation_match": bool(ar.get("citation", {}).get("match")), "citation_verified": all(bool(item.get("ok")) for item in verification) if verification else None, "pass1_block_present": block_present, "ghost_answer": is_ghost, "ghost_answer_with_value": is_ghost_with_value, "false_absent": is_false_absent})
                for dimension, value in (("source_run", str(case["source_run_id"])), ("question_key", str(question.get("key"))), ("question_type", str(question.get("question_type", ""))), ("group_size", str(len(questions)))):
                    bucket = report["breakdown"][dimension].setdefault(value, {"total": 0, "correct": 0, "excluded": 0})
                    if valid_cand:
                        bucket["total"] += 1; bucket["correct"] += int(ar["match"])
                    else:
                        bucket["excluded"] += 1
            if cand:
                report["cost"] += float(cand.get("cost") or 0)
                usage = _loads(cand.get("usage_json"), {}) or {}
                report["tokens"]["input"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                report["tokens"]["output"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                report["tokens"]["cached"] += int(usage.get("cached_tokens") or 0)
                if cand.get("latency_ms") is not None:
                    report["latency_ms"]["total"] += int(cand["latency_ms"]); report["latency_ms"]["count"] += 1
    for report in reports.values():
        report["group_accuracy"] = report["group_correct"] / report["eligible_groups"] if report["eligible_groups"] else None
        report["answer_accuracy"] = report["answer_correct"] / report["answer_total"] if report["answer_total"] else None
        for component in _COMPONENTS:
            entry = report["components"][component]
            entry["rate"] = entry["correct"] / entry["total"] if entry["total"] else None
        report["schema_failure_reasons"] = dict(report["schema_failure_reasons"])
        count = report["latency_ms"]["count"]
        report["latency_ms"]["average"] = report["latency_ms"]["total"] / count if count else None
        # Headline reference-free fabrication/data-loss rates (see the
        # ``ghost_answers``/``false_absent`` accumulation above).
        report["ghost_rate"] = report["ghost_answers"] / report["pass1_blocks_absent"] if report["pass1_blocks_absent"] else 0.0
        report["false_absent_rate"] = report["false_absent"] / report["pass1_blocks_present"] if report["pass1_blocks_present"] else 0.0
    return {"dataset": dataset, "reference": ref_row["name"], "quote_threshold": quote_threshold, "excluded": dict(excluded), "models": reports, "rows": rows, "answer_rows": answer_rows}


def write_report(report: Mapping[str, Any], json_path: str | Path | None = None, csv_path: str | Path | None = None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if csv_path:
        rows = list(report.get("rows", []))
        with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
            fields = ["candidate", "case_key", "source_run", "group_size", "group_match", "candidate_status", "source_text_kind",
                      "pass1_blocks_present", "pass1_blocks_absent", "ghost_answers", "ghost_answers_with_value", "false_absent"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({k: row.get(k) for k in fields} for row in rows)


compare = evaluate
evaluate_results = evaluate
