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
{options_section}"""


def _format_options(question: Question) -> str:
    if not question.options:
        return ""
    lines = ["Valid options (use the 'value' string exactly):"]
    for opt in question.options:
        lines.append(f"  {opt.value!r} — {opt.label}")
    if question.allow_multiple:
        lines.append("Multiple selections allowed.")
    return "\n".join(lines)


def build_messages(
    source_text: str,
    questions: list[Question],
    *,
    text_source: str = "full_text",
    system_prompt: str | None = None,
    cache_first: str = "text",
) -> list[dict]:
    """Return messages=[system, user_prefix, user_questions].

    The prefix message is kept separate so caching.py can mark it.
    Callers should concatenate or pass the full list.
    """
    source_label = "Full paper text (OCR)" if text_source == "full_text" else "Abstract"
    prefix_content = (
        f"=== {source_label} ===\n\n{source_text}\n\n=== END OF PAPER TEXT ==="
    )

    q_blocks = []
    for q in questions:
        help_section = f"Help: {q.help_text}" if q.help_text else ""
        options_section = _format_options(q)
        q_blocks.append(
            _QUESTION_BLOCK_TEMPLATE.format(
                key=q.key,
                label=q.label,
                help_section=help_section,
                options_section=options_section,
            )
        )

    question_content = (
        "Answer each question below for the paper text provided above.\n\n"
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


def build_format_messages(
    pass1_text: str,
    questions: list[Question],
) -> list[dict]:
    """Pass-2 messages: restructure pass-1 output into typed JSON.

    Always requests {"results": [...]} format. When called with
    response_format=json_object, the API enforces valid JSON; otherwise the
    model is still guided toward the same structure and parse_structured_output
    falls back gracefully if needed.
    """
    q_specs = []
    for q in questions:
        spec = f"key={q.key!r} type={q.question_type}"
        if q.question_type == "categorical":
            vals = [opt.value for opt in q.options]
            spec += f" allow_multiple={q.allow_multiple} valid_values={vals!r}"
        q_specs.append(spec)

    content = f"""\
Below is a free-form annotation output. Your only job is to faithfully restructure it into JSON.
Do NOT rephrase, shorten, or summarise anything — copy text verbatim.

{pass1_text}

Return a JSON object with a single "results" key containing one entry per question:
{{"results": [
  {{"key": "<key>", "value": <typed_value>, "cited_text": "<verbatim quote or empty>", "comment": "<verbatim reasoning>", "confidence": <0-20 or null>}},
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
Output entries in the same order as the questions listed above.
"""
    return [{"role": "user", "content": content}]
