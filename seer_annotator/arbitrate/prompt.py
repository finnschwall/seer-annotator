"""Build LLM message lists for arbitration (Pass-1 dispute adjudication).

Pass-2 (formatting) is intentionally *not* duplicated here — annotate.prompt's
build_format_messages, annotate.parse's parse_structured_output/_RESPONSE_FORMAT,
and annotate.verify's verify_citation are reused unchanged, because Pass-1's output
here follows the same "--- ANSWER: <key> ---" template as annotation's.
"""

from __future__ import annotations

from ..config import Candidate, Question

DEFAULT_ARBITER_SYSTEM = """\
You are an evidence-based adjudicator for a systematic review. Work through the \
questions in the order shown. Some are disagreements; others are continuation \
questions that one or more raters did not reach after excluding the paper. Independently \
decide each applicable answer by reading the evidence.

Rules:
- Weigh evidence, not confidence or verbosity. A short, correct comment beats a long, \
wrong one.
- You are not restricted to the candidates' proposed values. If the source text (when \
provided) supports a different value than any candidate proposed, use that value \
instead — state why in your reasoning.
- If an inclusion-criteria answer excludes the paper, STOP: do not answer any later \
questions. Likewise, do not answer a question whose listed conditions are not met.
- If the source text genuinely does not contain enough information to decide \
confidently, say so explicitly and give a low confidence score rather than guessing. \
A flagged uncertain answer is better than a wrong confident one.

Answer every applicable question using this template (keep the field labels):

--- ANSWER: <key> ---
Quotes:
- "<verbatim span copied from the paper, or from a candidate's cited evidence if no paper text is given>"
- "<another verbatim span>"   (add more bullets only for genuinely separate passages)
Reasoning: <step-by-step reasoning weighing the candidates' evidence>
Answer: <your answer; must be one of the valid values if options are listed>
Confidence: <0-20>

Rules for the Quotes section:
- Each bullet is ONE continuous passage, copied verbatim and wrapped in double quotes.
- Use a separate bullet for each genuinely distinct, non-adjacent passage you rely on. Do not \
merge separate passages into one bullet.
- Within a single bullet you may use ' ... ' to mark words omitted from that one continuous \
passage. NEVER use ' ... ' to join two unrelated passages — use two bullets instead.
- Put ONLY verbatim text inside the double quotes. Do not paraphrase, summarise, or add your \
own words there.
- If the answer is not supported by any specific span, write exactly one bullet:
  - [NO DIRECT QUOTE]
  and give the basis for your answer in the Reasoning section instead.

Confidence scale: 0=pure guess/no support in the evidence, 10=genuine uncertainty, 20=answer is \
explicitly and unambiguously stated in the text.
"""


def _format_candidates(candidates: list[Candidate], *, anonymize_raters: bool) -> str:
    if not candidates:
        return "(no candidates recorded)"
    lines: list[str] = []
    for i, c in enumerate(candidates):
        label = f"Candidate {chr(ord('A') + i)}" if anonymize_raters else c.rater_key
        lines.append(f"{label}:")
        if c.answer_status != "answered":
            reason = c.detail or c.answer_status.replace("_", " ")
            if c.stopped_at_question_key and c.stopped_at_question_key not in reason:
                reason = f"{reason} (at {c.stopped_at_question_key})"
            lines.append(f"  No answer: {reason}")
            continue
        lines.append(f"  Proposed value: {c.value!r}")
        if c.comment:
            lines.append(f"  Reasoning: {c.comment}")
        if c.cited_text:
            lines.append(f"  Cited evidence: {c.cited_text!r}")
        if not c.comment and not c.cited_text:
            lines.append("  (no reasoning or citation given)")
    return "\n".join(lines)


def _format_options(question: Question) -> str:
    if not question.options:
        return ""
    lines = ["Valid options (use the 'value' string exactly):"]
    for opt in question.options:
        lines.append(f"  {opt.value!r} — {opt.label}")
    if question.allow_multiple:
        lines.append("Multiple selections allowed.")
    return "\n".join(lines)


def _format_section_header(question: Question, prev_section_key: str | None) -> str:
    """Mirror of `annotate.prompt._format_section_header` for the adjudication prompt —
    same section-transition semantics, no cross-batch state. See that function's
    docstring for the batching caveat."""
    if question.section_key is None or question.section_key == prev_section_key:
        return ""
    lines = [f"=== SECTION: {question.section_label or question.section_key} ==="]
    if question.section_notes:
        lines.append(question.section_notes)
    return "\n".join(lines) + "\n\n"


_DISPUTE_BLOCK_TEMPLATE = """\
--- {item_heading}: {key} ---
Label: {label}
{help_section}
{ic_section}
{condition_section}
{options_section}
Candidates:
{candidates_section}"""


def build_dispute_messages(
    source_text: str | None,
    questions: list[Question],
    candidates_by_version_id: dict[int, list[Candidate]],
    *,
    anonymize_raters: bool = True,
    text_source: str = "full_text",
    system_prompt: str | None = None,
    cache_first: str = "text",
    item_types_by_version_id: dict[int, str] | None = None,
) -> list[dict]:
    """Return messages=[system, user_prefix, user_questions] for Pass-1 adjudication.

    Mirrors annotate.prompt.build_messages()'s shape so downstream Pass-2 formatting
    and parsing are reusable unchanged. When text_source == "candidates_only" (or no
    source text is available), no paper-text block is included at all — the
    adjudicator reasons only from the candidates' stated evidence.
    """
    q_blocks = []
    prev_section_key: str | None = None
    for q in questions:
        section_header = _format_section_header(q, prev_section_key)
        prev_section_key = q.section_key
        help_section = f"Help: {q.help_text}" if q.help_text else ""
        ic_section = (
            "This is an INCLUSION-CRITERIA question — an incorrect resolution can "
            "flip whether the paper is included in the review."
            if q.is_ic
            else ""
        )
        condition_section = ""
        if q.conditions:
            rendered = []
            for cond in q.conditions:
                negate = "not " if cond.get("negate") else ""
                required = cond.get("required_value") or "any non-empty answer"
                rendered.append(f"{cond.get('depends_on')} must {negate}equal {required!r}")
            condition_section = "Applicable only when: " + "; and ".join(rendered)
        options_section = _format_options(q)
        candidates_section = _format_candidates(
            candidates_by_version_id.get(q.version_id, []),
            anonymize_raters=anonymize_raters,
        )
        q_blocks.append(
            section_header
            + _DISPUTE_BLOCK_TEMPLATE.format(
                item_heading=(
                    "CONTINUATION QUESTION"
                    if (item_types_by_version_id or {}).get(q.version_id) == "continuation"
                    else "DISPUTED QUESTION"
                ),
                key=q.key,
                label=q.label,
                help_section=help_section,
                ic_section=ic_section,
                condition_section=condition_section,
                options_section=options_section,
                candidates_section=candidates_section,
            )
        )

    question_content = (
        "Adjudicate the ordered questions below. Stop after an excluding IC answer and "
        "omit condition-inactive questions.\n\n" + "\n\n".join(q_blocks)
    )

    if text_source == "candidates_only" or not source_text:
        prefix_content = (
            "No source paper text is provided for this adjudication. Decide based "
            "only on the candidates' stated values, reasoning, and cited evidence below."
        )
    else:
        source_label = "Full paper text (OCR)" if text_source == "full_text" else "Abstract"
        prefix_content = f"=== {source_label} ===\n\n{source_text}\n\n=== END OF PAPER TEXT ==="

    if cache_first == "questions":
        first, second = question_content, prefix_content
    else:
        first, second = prefix_content, question_content

    return [
        {"role": "system", "content": system_prompt or DEFAULT_ARBITER_SYSTEM},
        {"role": "user", "content": first},
        {"role": "user", "content": second},
    ]
