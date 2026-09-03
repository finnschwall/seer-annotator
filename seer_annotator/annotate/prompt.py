"""Build LLM message lists for annotation."""

from __future__ import annotations

from ..config import Question


# DEFAULT_SYSTEM = (
#     "You are a systematic review data extractor. "
#     "Answer each question strictly based on the provided paper text. "
#     "Do not hallucinate. You must always commit to an answer!\n\n"
#     "For each question:\n"
#     "1. Quote the most relevant verbatim span(s) from the paper that support your answer (or 'not found').\n"
#     "2. Reason step-by-step.\n"
#     "3. State your answer (must be one of the valid values if options are listed).\n"
#     "4. State confidence (0–20, where 0=pure guess/no support in the text, 10=genuine uncertainty, 20=answer is explicitly and unambiguously stated in the text)."
# )

DEFAULT_SYSTEM = """\
You are a systematic review data extractor. Answer each question strictly based on the \
provided paper text. Do not hallucinate. You must always commit to an answer!

Answer every question using this template (keep the field labels):

--- ANSWER: <key> ---
Quotes:
- "<verbatim span copied from the paper>"
- "<another verbatim span>"   (add more bullets only for genuinely separate passages)
Reasoning: <step-by-step reasoning>
Answer: <your answer; must be one of the valid values if options are listed>
Confidence: <0-20>

Rules for the Quotes section:
- Each bullet is ONE continuous passage, copied verbatim and wrapped in double quotes.
- Use a separate bullet for each genuinely distinct, non-adjacent passage you rely on. Do not \
merge separate passages into one bullet.
- Within a single bullet you may use ' ... ' to mark words omitted from that one continuous \
passage. NEVER use ' ... ' to join two unrelated passages — use two bullets instead. A bullet \
boundary means 'a different part of the paper'; ' ... ' means 'a skip within the same passage'.
- Put ONLY verbatim text inside the double quotes. Do not paraphrase, summarise, or add your \
own words there.
- If the answer is not supported by any specific span — e.g. it is a general finding across \
the paper, or it is inferred (such as the language the paper is written in) — write exactly \
one bullet:
  - [NO DIRECT QUOTE]
  and give the basis for your answer in the Reasoning section instead. Never put explanatory \
(non-verbatim) text inside the Quotes section.

Confidence scale: 0=pure guess/no support in the text, 10=genuine uncertainty, 20=answer is \
explicitly and unambiguously stated in the text.
"""

_QUESTION_BLOCK_TEMPLATE = """\
--- QUESTION: {key} ---
Label: {label}
{help_section}
{condition_section}
{options_section}{ic_section}"""

# Standing instruction added to the questions message only when
# RunConfig.early_exit_on_ic_exclusion is on. Short and placed once (not
# per-question) since it's a cached prefix — negligible added input cost. The
# savings this buys are truncated Pass-1 (and downstream Pass-2) *output*;
# correctness never depends on the model actually obeying it — the worker
# recomputes the exclusion point deterministically after Pass-2 (see
# annotate/scope.py) and discards/relabels anything answered past it anyway.
_EARLY_EXIT_INSTRUCTION = """\
IMPORTANT — answer the questions below IN THE EXACT ORDER GIVEN. Some are \
marked as inclusion-criteria (IC) gates above. The moment your answer to an IC \
question EXCLUDES the paper (see that question's "Excludes if" line), STOP \
immediately — do not answer any further questions after it.
"""


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
    """Render a section header block when `question` starts a new section relative to
    the previous question in this same `build_messages` call, mirroring the section
    card header + notes human annotators see in the SEER UI (`Section.label`/`.notes`).

    Only fires on a section *transition* within the questions passed to one call —
    there's no cross-batch state, so if a section's questions get split across
    multiple LLM calls (e.g. per_question/size batching), each call re-renders the
    header for whichever of that section's questions it happens to start with.
    Returns "" if `question` has no section, or is a continuation of the same
    section as the previous question.
    """
    if question.section_key is None or question.section_key == prev_section_key:
        return ""
    lines = [f"=== SECTION: {question.section_label or question.section_key} ==="]
    if question.section_notes:
        lines.append(question.section_notes)
    return "\n".join(lines) + "\n\n"


def _format_ic_section(question: Question) -> str:
    """Render which value(s) mean "excludes the paper" for an IC question.

    Purely descriptive — always shown for is_ic questions regardless of
    early_exit_on_ic_exclusion, so the model has the semantics in front of it
    even when the worker isn't asking it to stop early. Uses ic_passes
    (categorical) / ic_include_when_true (boolean), mirroring
    annotations.ic.ic_answer_passes on the SEER side.
    """
    if not question.is_ic:
        return ""
    if question.question_type == "boolean":
        if question.ic_include_when_true is None:
            return ""
        excluding = not question.ic_include_when_true
        return f"\nThis is an INCLUSION-CRITERIA question. Excludes if: {excluding!s}"
    if question.question_type == "categorical":
        excluding = [opt.value for opt in question.options if opt.ic_passes is not None and not opt.ic_passes]
        if not excluding:
            return ""
        return (
            "\nThis is an INCLUSION-CRITERIA question. Excludes if the answer is: "
            + ", ".join(repr(v) for v in excluding)
        )
    return ""


def _format_conditions(question: Question) -> str:
    if not question.conditions:
        return ""
    rendered = []
    for cond in question.conditions:
        negate = "not " if cond.get("negate") else ""
        required = cond.get("required_value") or "any non-empty answer"
        rendered.append(f"{cond.get('depends_on')} must {negate}equal {required!r}")
    return "Answer only when: " + "; and ".join(rendered)


def build_messages(
    source_text: str,
    questions: list[Question],
    *,
    text_source: str = "full_text",
    system_prompt: str | None = None,
    cache_first: str = "text",
    early_exit_on_ic_exclusion: bool = False,
) -> list[dict]:
    """Return messages=[system, user_prefix, user_questions].

    The prefix message is kept separate so caching.py can mark it.
    Callers should concatenate or pass the full list.

    ``early_exit_on_ic_exclusion`` mirrors ``RunConfig.early_exit_on_ic_exclusion``
    (default False, i.e. unchanged prompt): when True, adds the standing
    "stop at the first IC exclusion" instruction. The per-question IC
    exclusion semantics (``_format_ic_section``) are rendered unconditionally
    for ``is_ic`` questions — that's just descriptive context, not a behavior
    change either way.
    """
    source_label = "Full paper text (OCR)" if text_source == "full_text" else "Abstract"
    prefix_content = (
        f"=== {source_label} ===\n\n{source_text}\n\n=== END OF PAPER TEXT ==="
    )

    q_blocks = []
    prev_section_key: str | None = None
    for q in questions:
        section_header = _format_section_header(q, prev_section_key)
        prev_section_key = q.section_key
        help_section = f"Help: {q.help_text}" if q.help_text else ""
        options_section = _format_options(q)
        ic_section = _format_ic_section(q)
        condition_section = _format_conditions(q)
        q_blocks.append(
            section_header
            + _QUESTION_BLOCK_TEMPLATE.format(
                key=q.key,
                label=q.label,
                help_section=help_section,
                condition_section=condition_section,
                options_section=options_section,
                ic_section=ic_section,
            )
        )

    question_content = (
        "Answer each applicable question below for the paper text provided above. "
        "Work in order and omit questions whose listed conditions are not met.\n\n"
        + ("\n" + _EARLY_EXIT_INSTRUCTION + "\n" if early_exit_on_ic_exclusion else "")
        + "\n\n".join(q_blocks)
    )

    if cache_first == "questions":
        first, second = question_content, prefix_content
    else:
        first, second = prefix_content, question_content

    return [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM},
        {"role": "user", "content": first},
        {"role": "user", "content": second},
    ]


_STATUS_SCHEMA_LINE = (
    '  {"key": "<key>", "value": <typed_value>, "cited_text": "<verbatim quote or empty>", '
    '"comment": "<verbatim reasoning>", "confidence": <0-20 or null>, "status": "<ok|absent|unmappable>"},'
)

_NO_STATUS_SCHEMA_LINE = (
    '  {"key": "<key>", "value": <typed_value>, "cited_text": "<verbatim quote or empty>", '
    '"comment": "<verbatim reasoning>", "confidence": <0-20 or null>},'
)

# Purely mechanical — this is Pass-2's ONE job beyond restructuring: report
# whether each question's block is present in the text at all, by inspecting
# the text only. It must NEVER reason about inclusion criteria/scope (that's
# the worker's job, see annotate/scope.py) and NEVER guess a value or explain
# *why* something is absent.
_STATUS_RULE = """\
- status: purely mechanical, based only on what's present in the annotation text above —
    * "ok"         — the question's block is present and you extracted a value from it
                      (a null value for a genuinely not-determinable answer still counts as "ok").
    * "absent"     — there is NO "--- ANSWER: <key> ---" block for this question anywhere
                      in the text above.
    * "unmappable" — the block IS present, but its content cannot be expressed as a valid
                      value for this question's type/options (e.g. free text where a
                      categorical value was required).
  Never guess a value to avoid "absent"/"unmappable", and never explain *why* a value is
  absent or unmappable — just report which of the three applies.
  For "absent"/"unmappable" entries, cited_text and comment must be "" (null is also accepted).
"""


def build_format_messages(
    pass1_text: str,
    questions: list[Question],
    *,
    require_status: bool = False,
) -> list[dict]:
    """Pass-2 messages: restructure pass-1 output into typed JSON.

    Always requests {"results": [...]} format. When called with
    response_format=json_object, the API enforces valid JSON; otherwise the
    model is still guided toward the same structure and parse_structured_output
    falls back gracefully if needed.

    ``require_status`` adds the mechanical ok/absent/unmappable per-entry
    status to the prompt's JSON contract (paired with
    ``annotate.parse.ANNOTATE_RESPONSE_FORMAT`` for structured-output mode).
    Default False keeps the prompt text byte-for-byte identical to before —
    this function is also reused unchanged by arbitration (see
    ``arbitrate/prompt.py``'s module docstring), which must NOT gain the
    status field.
    """
    q_specs = []
    for q in questions:
        spec = f"key={q.key!r} type={q.question_type}"
        if q.question_type == "categorical":
            vals = [opt.value for opt in q.options]
            spec += f" allow_multiple={q.allow_multiple} valid_values={vals!r}"
        q_specs.append(spec)

    schema_line = _STATUS_SCHEMA_LINE if require_status else _NO_STATUS_SCHEMA_LINE
    status_rule = _STATUS_RULE if require_status else ""

    content = f"""\
Below is a free-form annotation output. Your only job is to faithfully restructure it into JSON.
Do NOT rephrase, shorten, or summarise anything — copy text verbatim.

{pass1_text}

Return a JSON object with a single "results" key containing one entry per question:
{{"results": [
{schema_line}
  ...
]}}

Questions:
{chr(10).join(q_specs)}

Rules:
- value: for categorical, must be exactly one of valid_values (or a list if allow_multiple=true). For boolean, true or false (JSON). For text/integer/float, a string. If not determinable, null.
- cited_text: copy VERBATIM from the annotation's Quotes section — do not rephrase, shorten, or paraphrase.
  Each bullet in the Quotes section is one separate passage:
    * exactly one bullet  -> a single string.
    * two or more bullets -> a JSON array of strings, one element per bullet.
  A " ... " inside a single bullet is part of that one passage — keep it inside the string; it is
  NOT a separator and must NOT trigger an array. Only bullet count decides single-vs-array.
  If the Quotes section is "[NO DIRECT QUOTE]" (or otherwise contains no verbatim quote), also just copy that!
- comment: copy VERBATIM from the annotation — do not rephrase, shorten, or paraphrase.
- confidence: copy the integer (0–20) exactly as written. If not given or not determinable, null.
{status_rule}Output entries in the same order as the questions listed above.
"""
    return [{"role": "user", "content": content}]
