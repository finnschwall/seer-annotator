"""Ghost-answer and reliability analysis for a Phase-2 benchmark database.

Reference-free. Reads only stored executions/answers plus each case's frozen
pass1_text, so it never calls a model. Two questions:

  1. Fabrication — how often does a Pass-2 model report a usable status for a
     question Pass-1 never answered, and does that change the IC gate outcome?
  2. Reliability — which models are safe to run as the Pass-2 formatter.

Usage: python benchmarking/bench500/analyze.py <db> <dataset>
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from seer_annotator.annotate.parse import pass1_block_present, parse_structured_output_diagnostic
from seer_annotator.annotate.scope import compute_exclusion_index
from seer_annotator.benchmarking.runner import _question


# --- citation fidelity against Pass-1 -------------------------------------
# Reference-free and the most decision-relevant measure here: does the model
# preserve the quoted text Pass-1 selected? Separates TEXT LOSS (a quote's words
# are gone) from FORMAT-ONLY divergence (all text present, but bullets joined
# into one string instead of the required array). Those are very different
# severities and the reference-agreement score conflates them.
_HDR_RE = re.compile(
    r"^[ \t]*[-*=]{2,}[ \t]*ANSWER[ \t]*:[ \t]*([A-Za-z0-9_.\-<>]+)[ \t]*[-*=]*[ \t]*$",
    re.I | re.M)


def pass1_block_body(text: str, key: str) -> str | None:
    """The body of *key*'s Pass-1 block, or None. Requires a decorated header on
    its own line — a bare "Answer: Yes" line must never be mistaken for one."""
    m = re.search(
        r"^[ \t]*[-*=]{2,}[ \t]*ANSWER[ \t]*:[ \t]*" + re.escape(key)
        + r"[ \t]*[-*=]*[ \t]*$", text, re.I | re.M)
    if not m:
        return None
    rest = text[m.end():]
    nxt = _HDR_RE.search(rest)
    return rest[:nxt.start()] if nxt else rest


def quote_bullets(body: str) -> list[str]:
    m = re.search(r"Quotes:\s*\n(.*?)(?=\n[ \t]*Reasoning:)", body, re.S | re.I)
    if not m:
        return []
    out = [l.strip()[1:].strip().strip('"') for l in m.group(1).splitlines()
           if l.strip().startswith("-")]
    return [x for x in out if x]


def _cmp(s: object) -> str:
    """Fold to comparable form. NFKD so a curly apostrophe matches a straight one."""
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(s or "")).casefold())


DB, DATASET = sys.argv[1], sys.argv[2]
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

CLAIMS_PRESENT = {"ok", "unmappable"}   # statuses asserting the block exists

cases = {
    r["id"]: r for r in con.execute(
        "SELECT c.* FROM cases c JOIN datasets d ON d.id=c.dataset_id WHERE d.name=?", (DATASET,))
}
qcache: dict[int, list] = {}


def questions_for(case_id: int) -> list:
    if case_id not in qcache:
        qcache[case_id] = [_question(q) for q in json.loads(cases[case_id]["questions_json"])]
    return qcache[case_id]


stats: dict[str, Counter] = defaultdict(Counter)
ghosts: dict[str, list] = defaultdict(list)
exec_states: dict[str, Counter] = defaultdict(Counter)
usage: dict[str, Counter] = defaultdict(Counter)
latencies: dict[str, list] = defaultdict(list)

for e in con.execute(
    """SELECT e.*, mc.name AS model FROM executions e
       JOIN model_configs mc ON mc.id=e.model_config_id
       JOIN cases c ON c.id=e.case_id JOIN datasets d ON d.id=c.dataset_id
       WHERE d.name=?""", (DATASET,)):
    m = e["model"]
    exec_states[m][e["status"]] += 1
    if e["status"] != "complete":
        continue
    case = cases[e["case_id"]]
    p1 = case["pass1_text"]
    qs = questions_for(e["case_id"])
    # Recompute rather than trusting diagnostics_json: stored diagnostics were
    # written by whichever parser version ran at the time, and an older
    # validator wrongly flagged null cited_text/comment on absent entries as
    # schema violations. Recomputing keeps the comparison version-independent.
    diag = parse_structured_output_diagnostic(
        e["raw_response"] or "", [q.key for q in questions_for(e["case_id"])], annotate_mode=True)
    u = json.loads(e["usage_json"] or "{}")
    for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
        usage[m][k] += u.get(k) or 0
    usage[m]["cost_micro"] += int(round((e["cost"] or 0) * 1e6))
    if e["latency_ms"]:
        latencies[m].append(e["latency_ms"])

    stats[m]["groups"] += 1
    stats[m]["native_json"] += bool(diag.get("native_json"))
    stats[m]["fallback"] += bool(diag.get("fallback_used"))
    stats[m]["repair"] += bool(diag.get("repair_used"))
    for f in ("missing_keys", "duplicate_keys", "unexpected_keys", "schema_violations", "missing_fields"):
        if diag.get(f):
            stats[m][f"groups_with_{f}"] += 1
            stats[m][f] += len(diag[f])

    # The model's claim is read from `diag`, i.e. re-derived from raw_response
    # with the current parser, NOT from the stored answers table. The two differ
    # for older rows: an earlier parser wrote status="absent" whenever it could
    # not extract a key, so a model that emitted valid-but-awkward JSON looks
    # like it claimed "absent" when it did not. Attributing a parser limitation
    # to the model would rank models wrongly. `parse_recovered` counts that gap.
    stored_status = {
        r["key"]: r["mechanical_status"]
        for r in con.execute(
            "SELECT key, mechanical_status FROM answers WHERE execution_id=?", (e["id"],))
    }
    by_key = {a["key"]: a for a in diag["answers"]}
    raw_vals, enf_vals = {}, {}
    for q in qs:
        ans = by_key.get(q.key)
        if ans is None:
            stats[m]["answer_row_missing"] += 1
            continue
        st = ans.get("status") or "ok"
        if stored_status.get(q.key) == "absent" and st != "absent":
            stats[m]["parse_recovered"] += 1
        present = pass1_block_present(p1, q.key)
        stats[m]["answers"] += 1
        stats[m]["present" if present else "absent_in_p1"] += 1
        claims = st in CLAIMS_PRESENT
        if claims:
            raw_vals[q.key] = ans.get("value")
        if present and claims:
            enf_vals[q.key] = ans.get("value")
        if not present and claims:
            stats[m]["ghost"] += 1
            if ans.get("value") is not None:
                stats[m]["ghost_with_value"] += 1
            ghosts[m].append((case["case_key"], q.key, st, ans.get("value"), ans.get("confidence")))
        if present and st == "absent":
            stats[m]["false_absent"] += 1

        if present and st == "ok":
            want = quote_bullets(pass1_block_body(p1, q.key) or "")
            if want and want != ["[NO DIRECT QUOTE]"]:
                ct = ans.get("cited_text")
                flat = " ".join(str(x) for x in ct) if isinstance(ct, list) else str(ct or "")
                got = len(ct) if isinstance(ct, list) else (1 if ct else 0)
                stats[m]["cit_scored"] += 1
                if any(_cmp(w)[:60] not in _cmp(flat) for w in want):
                    stats[m]["cit_text_lost"] += 1
                elif got != len(want):
                    stats[m]["cit_format_only"] += 1
                else:
                    stats[m]["cit_perfect"] += 1

    raw_idx = compute_exclusion_index(qs, raw_vals)
    enf_idx = compute_exclusion_index(qs, enf_vals)
    if raw_idx != enf_idx:
        stats[m]["ic_gate_changed"] += 1
        ghosts[m].append(("IC-GATE", case["case_key"],
                          f"raw={qs[raw_idx].key if raw_idx is not None else None}",
                          f"enforced={qs[enf_idx].key if enf_idx is not None else None}", ""))


def pct(a, b):
    return f"{a / b:6.2%}" if b else "     —"


order = sorted(stats, key=lambda m: (stats[m]["ghost"], stats[m]["false_absent"]))
print(f"dataset: {DATASET}   cases: {len(cases)}\n")
print("FABRICATION — a status of ok/unmappable on a question Pass-1 never answered")
print(f"{'model':28s} {'done':>5s} {'answers':>8s} {'no P1 blk':>10s} {'ghost':>6s} {'rate':>7s} "
      f"{'w/value':>8s} {'IC gate moved':>13s}")
for m in order:
    s = stats[m]
    print(f"{m:28s} {exec_states[m]['complete']:5d} {s['answers']:8d} {s['absent_in_p1']:10d} "
          f"{s['ghost']:6d} {pct(s['ghost'], s['absent_in_p1'])} {s['ghost_with_value']:8d} "
          f"{s['ic_gate_changed']:13d}")

print("\nDATA LOSS — 'absent' claimed although the Pass-1 block is present")
print(f"{'model':28s} {'P1 blk present':>15s} {'false absent':>13s} {'rate':>7s} {'parse-recovered':>16s}")
for m in order:
    s = stats[m]
    print(f"{m:28s} {s['present']:15d} {s['false_absent']:13d} {pct(s['false_absent'], s['present'])} "
          f"{s['parse_recovered']:16d}")

print("\nWIRE RELIABILITY (per group of questions)")
print(f"{'model':28s} {'groups':>7s} {'native':>7s} {'fallbk':>7s} {'repair':>7s} {'missKey':>8s} "
      f"{'dupKey':>7s} {'unexKey':>8s} {'schemaV':>8s} {'errors':>7s}")
for m in order:
    s = stats[m]
    print(f"{m:28s} {s['groups']:7d} {pct(s['native_json'], s['groups'])} {s['fallback']:7d} "
          f"{s['repair']:7d} {s['missing_keys']:8d} {s['duplicate_keys']:7d} "
          f"{s['unexpected_keys']:8d} {s['schema_violations']:8d} {exec_states[m]['error']:7d}")

print("\nCITATION FIDELITY vs Pass-1 (reference-free; blocks that carry real quotes)")
print(f"{'model':28s} {'scored':>7s} {'perfect':>8s} {'format only':>12s} {'TEXT LOST':>10s} {'text intact':>12s}")
for m in sorted(stats, key=lambda x: stats[x]["cit_text_lost"]):
    s_ = stats[m]
    print(f"{m:28s} {s_['cit_scored']:7d} {s_['cit_perfect']:8d} {s_['cit_format_only']:12d} "
          f"{s_['cit_text_lost']:10d} {pct(s_['cit_scored'] - s_['cit_text_lost'], s_['cit_scored'])}")

print("\nCOST / SPEED (per completed group)")
print(f"{'model':28s} {'in tok':>8s} {'out tok':>8s} {'reason':>8s} {'$/1k grp':>9s} {'p50 ms':>8s} {'p95 ms':>8s}")
for m in order:
    g = max(1, stats[m]["groups"])
    lat = sorted(latencies[m]) or [0]
    print(f"{m:28s} {usage[m]['input_tokens']//g:8d} {usage[m]['output_tokens']//g:8d} "
          f"{usage[m]['reasoning_tokens']//g:8d} {usage[m]['cost_micro']/g/1000:9.2f} "
          f"{lat[len(lat)//2]:8d} {lat[min(len(lat)-1, int(len(lat)*0.95))]:8d}")

for m in order:
    if ghosts[m]:
        print(f"\n--- {m}: {len(ghosts[m])} fabrication / gate events ---")
        for row in ghosts[m][:25]:
            print("   ", *row)
