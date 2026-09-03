"""Deterministic IC-gate scope enforcement — the worker's authoritative pass/fail logic.

Mirrors SEER's ``annotations.ic.ic_answer_passes`` (see SEER's
``docs/concepts/data-model.md`` § "IC decisions are derived"): given a group of
questions in study-master order and their Pass-2-extracted values, find the
first inclusion-criteria (``is_ic``) answer that fails its gate, and everything
ordered after it is out of scope. This recomputation is authoritative — it
never trusts Pass-1 to have actually stopped early (see prompt.py's standing
instruction, which is a cost optimization only, not a correctness dependency).

Pass-2 itself never reasons about IC/scope (see prompt.py's mechanical
"ok"/"absent"/"unmappable" status contract in parse.py) — this module is the
one place that combines "is this an IC question" + "what value did it get"
into a scope decision.
"""

from __future__ import annotations

import logging

from ..config import Question
from .parse import pass1_block_present

logger = logging.getLogger(__name__)


def option_ic_passes(question: Question, value: object) -> bool | None:
    """Whether a single selected categorical option value passes its question's
    IC gate. Returns None when indeterminate: the option isn't found among the
    question's ``options``, or the option's ``ic_passes`` was never set (not a
    gating option) — either way there's no signal to exclude on, so callers
    treat this the same as "no answer" rather than fabricating a False.
    """
    sval = str(value)
    for opt in question.options:
        if opt.value == sval:
            if opt.ic_passes is None:
                return None
            return bool(opt.ic_passes)
    return None


def ic_answer_passes(question: Question, value: object) -> bool | None:
    """Mirror of SEER's ``annotations.ic.ic_answer_passes`` for one (question,
    extracted value) pair, evaluated worker-side from the wire-contract fields
    (``ic_include_when_true`` for boolean, per-option ``ic_passes`` for
    categorical).

    Returns True (passes/include), False (fails/exclude), or None
    (indeterminate: no value, an unmapped categorical value, or a question
    type unmappable question that isn't gate-relevant). Only meaningful when
    ``question.is_ic`` — callers must check that themselves; this function
    doesn't special-case non-IC questions beyond returning None for them.
    """
    if not question.is_ic:
        return None

    qt = question.question_type
    if qt == "boolean":
        if not isinstance(value, bool) or question.ic_include_when_true is None:
            return None
        return value == question.ic_include_when_true

    if qt == "categorical":
        if question.allow_multiple:
            if not isinstance(value, list) or not value:
                return None
            results = [option_ic_passes(question, v) for v in value]
            if any(r is None for r in results):
                return None
            return all(results)
        if value is None:
            return None
        return option_ic_passes(question, value)

    return None


def compute_exclusion_index(
    questions: list[Question], values: dict[str, object]
) -> int | None:
    """Find the first IC exclusion in *questions* order.

    ``values`` maps ``question.key -> extracted value`` for questions that
    have a determinable value at all (typically: Pass-2 status == "ok").
    Questions absent from ``values`` (no value / status not "ok") carry no
    exclusion signal — ``ic_answer_passes`` would return None for them too,
    so omitting them from ``values`` entirely is equivalent and lets callers
    build ``values`` straight from "ok" entries only.

    Returns the list index (into *questions*) of the first question whose
    answer fails its IC gate, or None if no question in the sequence excludes.
    """
    for i, q in enumerate(questions):
        if not q.is_ic:
            continue
        if q.key not in values:
            continue
        if ic_answer_passes(q, values[q.key]) is False:
            return i
    return None


def apply_scope_and_status(
    questions: list[Question],
    parsed: list[dict],
    *,
    enabled: bool,
    pass1_text: str | None = None,
) -> tuple[list[dict], int | None]:
    """Combine Pass-2's mechanical per-entry ``status`` with the worker's own
    exclusion-point computation into one final ``extraction_status`` per
    question, per the A4 mapping:

      - ordered strictly after the exclusion point -> "skipped" (the P2 value
        is discarded regardless of what it was — moot either way).
      - Pass-2 "unmappable" -> "invalid".
      - Pass-2 "ok" -> "ok".
      - Pass-2 "absent" at/before the exclusion point (or when there is no
        exclusion) -> "absent", a genuine omission the caller must still treat
        as an error/gap — NOT skipped.

    ``questions`` and ``parsed`` must be the same length and in the same
    (study-master) order — the order IS the IC sequence, no separate ordering
    field exists on the wire.

    ``enabled=False`` (i.e. ``RunConfig.early_exit_on_ic_exclusion`` off)
    short-circuits exclusion-point computation entirely (returns ``excl_idx =
    None`` unconditionally) so no cell can ever come out "skipped" — this is
    what keeps behavior byte-for-byte unchanged for runs that don't opt in,
    even if the data genuinely contains an IC exclusion. Only the "ok"/
    "unmappable"/"absent" per-key relabeling (from A3/A5) still applies, since
    that's a parse-tolerance bugfix independent of the IC-gating feature.

    Returns ``(relabeled, excl_idx)`` — ``relabeled`` is a new list of dicts
    (originals are not mutated), each with a final ``extraction_status`` key
    (one of "ok"/"invalid"/"absent"/"skipped") and, for "skipped" entries,
    ``value`` forced to ``None`` and ``extraction_detail`` naming the
    excluding question. ``excl_idx`` is the exclusion index (or None) so
    callers doing multi-round/early-exit batching can carry the decision
    forward to later groups/cells for the same paper.

    ``pass1_text``, when not ``None``, enables a DEMOTE-ONLY correctness
    guarantee: a Pass-2 entry may only keep a status that claims its answer
    block exists ("ok" or "unmappable") if Pass-1's free-form text actually
    contains an answer block for that question — either an
    ``--- ANSWER: <key> ---`` header, or a ``--- QUESTION: <key> ---`` header
    over a block that holds an ``Answer:`` field (see
    ``parse.pass1_block_present`` for why the second shape counts). This defends against a Pass-2 model that
    fabricates a value for a question Pass-1 never answered (observed: a weak
    local model copying another question's quotes/comment and stamping
    confidence 20 on them). Code can never promote "absent" -> "ok" — it has
    no way to invent a value that isn't in the text — so this only ever
    demotes a fabricated "ok"/"unmappable" down to "absent". The default
    ``None`` performs no check at all, which is what keeps every existing
    caller/test byte-for-byte unchanged, and is also what guarantees
    arbitration (whose Pass-1 text has no such answer blocks) is never
    affected — arbitration call sites simply never pass this parameter.
    """
    assert len(questions) == len(parsed), "questions/parsed must be aligned 1:1"

    if pass1_text is not None:
        # Enforce BEFORE the values_by_key loop below, which feeds
        # compute_exclusion_index: a fabricated "ok" must never be allowed to
        # move the IC exclusion point, even transiently.
        demoted: list[dict] = []
        for r in parsed:
            key = r.get("key")
            status = r.get("status", "ok")
            if status in ("ok", "unmappable") and not pass1_block_present(pass1_text, key):
                logger.warning(
                    "Pass-2 claimed status=%r for key=%r but pass-1 has no answer "
                    "block for it — demoting to 'absent'",
                    status, key,
                )
                r = dict(r)
                r["status"] = "absent"
                r["value"] = None
                r["cited_text"] = ""
                r["comment"] = ""
                r["confidence"] = None
                r["extraction_detail"] = f"pass-1 has no answer block for {key!r}"
            demoted.append(r)
        parsed = demoted

    def condition_matches(value: object, required: str, negate: bool) -> bool:
        if required == "":
            matched = value is not None and value != "" and value != []
        elif isinstance(value, bool):
            matched = str(value).lower() == required.lower()
        elif isinstance(value, list):
            matched = required in {str(v) for v in value}
        else:
            matched = str(value) == required
        return not matched if negate else matched

    values_by_key: dict[str, object] = {}
    condition_inactive: set[int] = set()
    for i, (q, r) in enumerate(zip(questions, parsed)):
        inactive = q.conditions and any(
            cond.get("depends_on") in values_by_key
            and not condition_matches(
                values_by_key[cond.get("depends_on")], cond.get("required_value", ""),
                bool(cond.get("negate")),
            )
            for cond in q.conditions
        )
        if inactive:
            condition_inactive.add(i)
        elif r.get("status", "ok") == "ok":
            values_by_key[q.key] = r.get("value")

    excl_idx: int | None = None
    if enabled:
        excl_idx = compute_exclusion_index(questions, values_by_key)
    excluding_key = questions[excl_idx].key if excl_idx is not None else None

    out: list[dict] = []
    for i, (q, r) in enumerate(zip(questions, parsed)):
        r = dict(r)
        if excl_idx is not None and i > excl_idx:
            r["value"] = None
            r["extraction_status"] = "skipped"
            r["extraction_detail"] = f"gated out: {excluding_key}=false"
        elif i in condition_inactive:
            r["value"] = None
            r["extraction_status"] = "skipped"
            r["extraction_detail"] = "condition not met"
        else:
            p2_status = r.get("status", "ok")
            if p2_status == "unmappable":
                r["extraction_status"] = "invalid"
            elif p2_status == "absent":
                r["extraction_status"] = "absent"
            else:
                r["extraction_status"] = "ok"
        out.append(r)

    return out, excl_idx
