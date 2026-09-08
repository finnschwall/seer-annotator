"""Parse pass-2 formatted output into per-question typed values."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Matches a bare JSON primitive in the value field — never a quoted string in our schema,
# so this is unambiguous even when the rest of the line has unescaped quotes.
_VALUE_RE = re.compile(r'"value"\s*:\s*(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

_PRIMITIVE_MAP = {"true": True, "false": False, "null": None}

# Mechanical per-entry status Pass-2 can report (annotate path only — see
# ANNOTATE_RESPONSE_FORMAT below). Purely about what's present in the text:
#   "ok"         — a value (possibly null for genuinely not-determinable) was
#                  extracted from the question's block.
#   "absent"     — no answer/block for this question is present in the text at all.
#   "unmappable" — content is present but cannot be expressed as a valid value
#                  for the type/options.
# Never IC/scope-aware — that judgment belongs to the worker (see annotate/scope.py).
_STATUS_VALUES = {"ok", "absent", "unmappable"}

# Structured-output JSON schema for Pass-2 (formatting). Shared by annotation and
# arbitration engines — both restructure free-form Pass-1 text into the same
# {key, value, cited_text, comment, confidence} shape per question/dispute.
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "annotation_results",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key":        {"type": "string"},
                            "value":      {"anyOf": [
                                {"type": "boolean"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ]},
                            "cited_text": {"anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]},
                            "comment":    {"type": "string"},
                            "confidence": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        },
                        "required": ["key", "value", "cited_text", "comment", "confidence"],
                    },
                },
            },
            "required": ["results"],
        },
    },
}


def _build_annotate_response_format() -> dict:
    """Extended Pass-2 schema variant used ONLY by the annotate path.

    Adds a required per-entry ``status`` (ok/absent/unmappable — see
    _STATUS_VALUES) to the shared ``_RESPONSE_FORMAT`` shape. Built as a deep
    copy so the shared constant (also imported by ``arbitrate/prompt.py``) is
    never mutated — arbitration keeps using ``_RESPONSE_FORMAT`` unchanged.
    """
    import copy

    fmt = copy.deepcopy(_RESPONSE_FORMAT)
    item_schema = fmt["json_schema"]["schema"]["properties"]["results"]["items"]
    item_schema["properties"]["status"] = {
        "type": "string",
        "enum": sorted(_STATUS_VALUES),
    }
    item_schema["required"].append("status")
    fmt["json_schema"]["name"] = "annotation_results_with_status"
    return fmt


# Built once at import time — the annotate path's Pass-2 response_format.
ANNOTATE_RESPONSE_FORMAT = _build_annotate_response_format()

# The fields every item must carry, read off the schema above rather than
# retyped. ``status`` is appended per call, because only the annotate path asks
# for it (see ``_build_annotate_response_format``) — an arbitration item that
# has no status is complete, not damaged.
_REQUIRED_ITEM_FIELDS = tuple(
    _RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["results"]["items"]["required"]
)


def _parse_raw_value(raw: str) -> object:
    """Convert a regex-captured primitive token to a Python value."""
    if raw in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[raw]
    try:
        return int(raw)
    except ValueError:
        return float(raw)


# A block header as the prompt template writes it: "ANSWER"/"QUESTION" in
# capitals, alone on its line apart from decoration. Deliberately
# case-SENSITIVE, unlike the key match below, so that a block's own
# "Answer: Yes" field line is never mistaken for the start of the next block.
_BLOCK_HEADER_LINE = re.compile(r"^[^A-Za-z0-9\n]{0,8}(?:ANSWER|QUESTION)\s*:", re.MULTILINE)

# The template's "Answer:" field label, tolerant of wrapping decoration
# ("**Answer:** yes"). Its presence is the evidence that a block holds a real
# answer rather than an echo of the question.
_ANSWER_FIELD_LINE = re.compile(r"^[^A-Za-z0-9\n]{0,4}Answer\s*:[^\S\n]*\S",
                                re.IGNORECASE | re.MULTILINE)


def _key_header_pattern(label: str, key: str) -> str:
    """Header pattern for *key* under *label* ("ANSWER" or "QUESTION").

    Unanchored on purpose: whatever decorates the header ("---", "**", "===")
    simply isn't matched, so no character class for it is needed. The negative
    lookahead (rather than ``\\b``) is what keeps key boundaries exact.
    """
    return label + r"\s*:\s*" + re.escape(key) + r"(?![A-Za-z0-9_.\-])"


def pass1_block_present(pass1_text: str, key: str) -> bool:
    """Whether Pass-1's free-form text contains an answer block for *key*
    (see prompt.py's standing instruction to the reasoning model).

    Two shapes count:

    1. An ``--- ANSWER: <key> ---`` header, on the header alone. The match is
       deliberately tolerant — case-insensitive, permissive about the
       dashes/asterisks/whitespace typically wrapping "ANSWER" and about
       whitespace around the colon — because a Pass-1 model that wobbles on
       the exact header formatting (e.g. "**ANSWER: key**" or
       "===ANSWER: key===") still genuinely answered the question. A strict
       match would turn that formatting wobble into silent data loss on an
       otherwise-good answer, which is worse than the fabrication problem
       this function exists to catch.

    2. A ``--- QUESTION: <key> ---`` header **whose block contains an
       "Answer:" field line**. Observed in production: a reasoning model
       answers correctly — real quotes, reasoning, answer, confidence — but
       labels the block after the question it is answering rather than after
       the answer, echoing the header shape the questions message uses. That
       is the same formatting wobble as (1), so it is recovered the same way.

    Shape 2 must show answer content; the bare header is not enough. Echoing
    a header is exactly what the model in the observed case did, so a model
    one step sloppier could echo a question header and then answer nothing
    (e.g. print the header, then stop on an IC exclusion) — and accepting a
    bare ``QUESTION:`` header would hand Pass-2 a free pass to invent that
    answer, which is precisely what this guard exists to prevent. Requiring
    the "Answer:" field leaves the accept-set of shape 1 unchanged: only the
    new fallback has to prove it holds an answer.

    Residual (accepted) gap: a Pass-1 text that echoes a question header for
    an unanswered key AND writes its other headers in lower case gives the
    block-boundary scan nothing to stop at, so the next block's "Answer:"
    line can be read as this block's. That needs mixed-case headers in one
    output, and still needs Pass-2 to fabricate on top of it.

    What this must NOT be tolerant about is key boundaries: it uses a
    negative lookahead (rather than ``\\b``) so a text containing only
    ``q10``'s block is never mistaken for containing ``q1``'s.
    """
    if re.search(_key_header_pattern("ANSWER", key), pass1_text, re.IGNORECASE):
        return True
    for m in re.finditer(_key_header_pattern("QUESTION", key), pass1_text, re.IGNORECASE):
        rest = pass1_text[m.end():]
        boundary = _BLOCK_HEADER_LINE.search(rest)
        block = rest[: boundary.start()] if boundary else rest
        if _ANSWER_FIELD_LINE.search(block):
            return True
    return False


class ExtractionError(RuntimeError):
    """Raised when pass-2 parsing fails to extract one or more question keys.

    Attributes:
        failed_keys: mapping of question key → parse_error message
    """

    def __init__(self, failed_keys: dict[str, str]) -> None:
        self.failed_keys = failed_keys
        keys_str = ", ".join(f"{k!r}: {v}" for k, v in failed_keys.items())
        super().__init__(
            f"Pass-2 extraction failed for {len(failed_keys)} key(s): {keys_str}"
        )


def _try_repair(line: str) -> dict | None:
    """Attempt to repair a malformed JSON line using json_repair.

    Returns the parsed object only if:
      - json_repair produces a valid dict with a 'key' field, AND
      - the 'value' field in the repaired object matches what a simple regex
        reads from the raw line (guards against json_repair inventing or
        coercing a value).

    Returns None if repair fails or the value cross-check fails.
    """
    try:
        from json_repair import repair_json
    except ImportError:
        return None

    try:
        repaired_str = repair_json(line)
        obj = json.loads(repaired_str)
    except Exception:
        return None

    if not isinstance(obj, dict) or not obj.get("key"):
        return None

    # Cross-check: the repaired value must agree with the raw-line regex.
    # This catches json_repair silently inventing or coercing a value.
    m = _VALUE_RE.search(line)
    if m:
        raw_value = _parse_raw_value(m.group(1))
        if obj.get("value") != raw_value:
            logger.error(
                "json_repair value mismatch — rejecting repair. "
                "raw line value=%r, repaired value=%r | line: %.200r",
                raw_value, obj.get("value"), line,
            )
            return None
    else:
        # No primitive value found in the raw line at all — if repair invented
        # a non-null value, that's suspicious; reject it.
        if obj.get("value") is not None:
            logger.error(
                "json_repair invented a value where raw line had none — rejecting. "
                "repaired value=%r | line: %.200r",
                obj.get("value"), line,
            )
            return None

    logger.warning(
        "Pass-2 JSON repaired (structural fix applied) for key=%r | line: %.200r",
        obj.get("key"), line,
    )
    return obj


def _try_repair_document(text: str) -> list | None:
    """Attempt a whole-document json_repair when json.loads fails on the full Pass-2 text.

    Handles failures the line-based fallback can't, e.g. pretty-printed multi-line
    JSON that's missing only the final closing brace — no single line is a complete
    JSON object there, so the line parser recovers nothing and every question gets
    silently placeholdered.

    _try_repair's per-line "value must match a raw-text regex" guard doesn't
    translate to a whole document (there's no single line to cross-check a given
    field against), so the guard here is coarser: accept the repair only if it
    yields at least one dict item with a non-empty "key". That's enough to reject
    json_repair inventing a plausible-looking structure from unrelated garbage,
    without trying to validate every field.

    Returns the item list (from {"results": [...]} or a bare [...]), or None if
    repair isn't available, fails, or doesn't produce a usable shape.
    """
    try:
        from json_repair import repair_json
    except ImportError:
        return None

    try:
        repaired = json.loads(repair_json(text))
    except Exception:
        return None

    if isinstance(repaired, dict) and "results" in repaired:
        items = repaired["results"]
    elif isinstance(repaired, list):
        items = repaired
    else:
        return None

    if not isinstance(items, list) or not any(
        isinstance(item, dict) and item.get("key") for item in items
    ):
        return None
    return items


_NO_DIRECT_QUOTE = "[NO DIRECT QUOTE]"


def cited_text_violation(raw: object) -> str | None:
    """Describe why ``raw`` is not a usable cited_text, or None if it is fine.

    Valid shapes are: absent/null, a string, or a list of strings. Anything else
    means the wire value did not survive intact — in practice a json_repair
    resynchronisation that swallowed the item's later fields into a nested list
    inside cited_text. The quote list is then not a quote list any more, and the
    rest of the item (comment, confidence, status) is missing, so the answer must
    be reported as an extraction failure rather than saved as a normal answer.
    """
    if raw is None or isinstance(raw, str):
        return None
    if isinstance(raw, list):
        bad = sorted({type(e).__name__ for e in raw if not isinstance(e, str)})
        if bad:
            return (
                f"cited_text list contains non-string element(s) of type "
                f"{', '.join(bad)} — the JSON item did not survive parsing intact: "
                f"{raw!r:.300}"
            )
        return None
    return (
        f"cited_text must be a string or a list of strings, got "
        f"{type(raw).__name__}: {raw!r:.300}"
    )


def _incomplete_item_violation(
    obj: dict, *, require_status: bool = False,
) -> str | None:
    """Describe why this item was never finished, or None if it looks complete.

    Only meaningful for an item read out of a **repaired** document. json_repair
    closes a reply that stopped early, so the document parses and every field the
    model reached is intact — but the fields it never reached are simply absent,
    and absent fields have defaults. ``comment`` defaults to "", ``confidence``
    to None, and ``status`` to "ok", which is what turns a reply that stopped
    mid-item into an answer that reads as a good one.

    So: an item from salvage that is missing a field the request asked for was
    cut off, whatever the reply-level evidence says. This is the backstop for a
    reply that gives no such evidence — one truncated below the output ceiling,
    or one the provider reported nothing about (see ``reply_quality``).

    ``cited_text_violation`` is the same judgment for the case where the damage
    landed inside ``cited_text`` specifically; that one still runs, because it
    also fires on documents that needed no repair at all.
    """
    required = _REQUIRED_ITEM_FIELDS + (("status",) if require_status else ())
    missing = [f for f in required if f not in obj]
    if not missing:
        return None
    return (
        f"incomplete item — this entry is missing {', '.join(missing)} in a Pass-2 "
        "reply that only parsed after repair, so the model stopped writing partway "
        "through it. The fields that did arrive are the ones it reached first, not "
        "the ones it got right."
    )


def _normalize_cited_text(raw: object) -> str | list | None:
    """Return None when the LLM signalled no verbatim quote is available.

    Also guarantees the shape downstream code assumes: a str, a list of str, or
    None/"" — never a nested list, dict or number. Elements that are not strings
    are dropped here; ``cited_text_violation`` is what reports that they were
    there (see _extract_result).
    """
    if raw is None:
        return ""
    if isinstance(raw, list):
        quotes = [s for s in raw if isinstance(s, str)]
        if [s.strip() for s in quotes] == [_NO_DIRECT_QUOTE]:
            return None
        return quotes
    if isinstance(raw, str):
        return None if raw.strip() == _NO_DIRECT_QUOTE else raw
    return ""


def _extract_result(
    obj: dict, *, salvaged: bool = False, require_status: bool = False,
) -> dict:
    """Pull the standard fields out of a parsed JSON object.

    ``status`` is read defensively: older prompts/schemas (arbitration, or the
    annotate path before this field existed) never send it, and a model can in
    principle emit a stray value outside _STATUS_VALUES — either way this
    defaults to "ok" (a value was present), matching pre-existing behavior for
    every caller that doesn't look at ``status`` at all.

    That default is only safe for an item the model actually finished writing.
    ``salvaged=True`` says this item came out of a document that json_repair had
    to close, and there the missing field may be one the model never reached —
    so the completeness check below runs and the item is flagged instead. See
    ``_incomplete_item_violation``.
    """
    status = obj.get("status")
    if status not in _STATUS_VALUES:
        status = "ok"
    raw_cited = obj.get("cited_text")
    result = {
        "key": obj.get("key"),
        "value": obj.get("value"),
        "cited_text": _normalize_cited_text(raw_cited),
        "comment": obj.get("comment") or "",
        "confidence": obj.get("confidence"),
        "status": status,
    }
    violation = cited_text_violation(raw_cited)
    if violation:
        logger.error(
            "Malformed cited_text for key=%r — reporting as an extraction failure: %s",
            obj.get("key"), violation,
        )
        result["cited_text_error"] = violation
    if salvaged:
        incomplete = _incomplete_item_violation(obj, require_status=require_status)
        if incomplete:
            logger.error(
                "Incomplete item for key=%r — reporting as an extraction failure: %s",
                obj.get("key"), incomplete,
            )
            result["item_error"] = incomplete
    return result


def _fill_missing(
    results: dict[str, dict], question_keys: list[str], *, annotate_mode: bool = False
) -> list[dict]:
    """Fill in an entry for every question key P2 didn't return.

    ``annotate_mode=False`` (default — arbitration and any other non-annotate
    caller): unchanged legacy behavior — a missing key gets a ``parse_error``,
    which is how those callers' whole-group ExtractionError detection works.

    ``annotate_mode=True`` (the annotate path only): a missing key gets
    ``status="absent"`` instead of a ``parse_error`` — the mechanical "no
    answer/block present in the text" signal that annotate/scope.py's
    exclusion-point + per-key tolerance logic consumes. No ``parse_error`` is
    set, so this key alone never trips the legacy whole-group ExtractionError
    path for annotate call sites (see orchestrator.py's per-key handling).
    """
    out = []
    for k in question_keys:
        if k in results:
            out.append(results[k])
        elif annotate_mode:
            out.append(
                {
                    "key": k,
                    "value": None,
                    "cited_text": "",
                    "comment": "",
                    "confidence": None,
                    "status": "absent",
                }
            )
        else:
            out.append(
                {
                    "key": k,
                    "value": None,
                    "cited_text": "",
                    "comment": "",
                    "confidence": None,
                    "parse_error": f"key {k!r} not found in pass-2 output",
                }
            )
    return out


def _strip_code_fence(text: str) -> str:
    """Remove a leading ```[json] ... ``` wrapper if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop the opening fence line and the closing ```
        lines = stripped.splitlines()
        end = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), None)
        if end is not None:
            return "\n".join(lines[1:end])
    return text


def parse_structured_output(
    text: str, question_keys: list[str], *, annotate_mode: bool = False,
    require_status: bool = False,
) -> list[dict]:
    """Parse pass-2 output when response_format=json_object was used.

    Expects {"results": [...]} or a bare [...] as the top-level JSON value.
    Falls back to parse_format_output() if the response is not valid JSON or
    does not match the expected shape (e.g. when drop_params silently removed
    response_format).

    ``annotate_mode`` is forwarded to ``_fill_missing`` — see its docstring.
    Default False preserves legacy behavior for every existing caller
    (arbitration, and annotate call sites that haven't opted in yet).

    ``require_status`` must match what the caller asked the model for (the
    ``require_status`` it passed to ``build_format_messages``). It is only used
    for the completeness check on a repaired document — see
    ``_incomplete_item_violation`` — and never changes what is parsed. It is a
    separate argument from ``annotate_mode`` because arbitration passes
    ``annotate_mode=True`` while asking for no status field at all.
    """
    text = _strip_code_fence(text)
    salvaged = False
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "results" in obj:
            items = obj["results"]
        elif isinstance(obj, list):
            items = obj
        else:
            raise ValueError(f"unexpected top-level JSON shape: {type(obj).__name__}")
    except (json.JSONDecodeError, ValueError) as exc:
        # Try a whole-document repair (handles e.g. pretty-printed JSON missing only
        # the final closing brace) before giving up on structured JSON entirely — the
        # line-based fallback below only understands JSONL and misses that case.
        items = _try_repair_document(text)
        if items is None:
            logger.warning(
                "parse_structured_output: response is not valid JSON (%s) — falling back "
                "to line-by-line parser. First 200 chars: %.200r", exc, text
            )
            return parse_format_output(
                text, question_keys, annotate_mode=annotate_mode,
                require_status=require_status,
            )
        logger.warning(
            "parse_structured_output: whole-document JSON repair recovered %d item(s) "
            "after %s. First 200 chars: %.200r", len(items), exc, text
        )
        # Everything read out of these items is salvage: the document parsed
        # only because json_repair closed what the model never wrote.
        salvaged = True

    results: dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and item.get("key"):
            r = _extract_result(item, salvaged=salvaged, require_status=require_status)
            results[r["key"]] = r

    return _fill_missing(results, question_keys, annotate_mode=annotate_mode)


def parse_format_output(
    text: str, question_keys: list[str], *, annotate_mode: bool = False,
    require_status: bool = False,
) -> list[dict]:
    """Extract one dict per question from pass-2 JSON-lines output (repair/opt-out path).

    Returns list of {key, value, cited_text, comment, confidence, status} dicts.
    Missing/malformed keys get value=None with either a parse_error note
    (default) or status="absent" (``annotate_mode=True`` — see _fill_missing).
    """
    results: dict[str, dict] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Pass-2 JSON parse error, attempting repair: %s | line: %.200r", exc, line
            )
            obj = _try_repair(line)
            if obj is None:
                logger.error(
                    "Pass-2 repair failed or rejected — line will be missing from results: %.200r",
                    line,
                )
                continue
            # Salvage is per line here, not per document: the other lines were
            # whole JSON objects and are trusted as written.
            salvaged = True
        else:
            salvaged = False

        key = obj.get("key")
        if key:
            r = _extract_result(obj, salvaged=salvaged, require_status=require_status)
            results[key] = r

    return _fill_missing(results, question_keys, annotate_mode=annotate_mode)


def parse_structured_output_diagnostic(
    text: str, question_keys: list[str], *, annotate_mode: bool = True
) -> dict:
    """Parse formatted output and retain mechanical parser diagnostics.

    This is deliberately an additive API: existing callers should continue to
    use :func:`parse_structured_output`, whose return shape and error semantics
    are unchanged.  ``present`` distinguishes an explicitly emitted
    ``status="absent"`` entry from a key which was omitted altogether.
    """
    expected = list(question_keys)
    expected_set = set(expected)
    stripped = _strip_code_fence(text)
    native_json = False
    fallback_used = False
    repair_used = False
    parse_errors: list[str] = []
    raw_items: list[dict] = []
    raw_entries: list[object] = []

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "results" in obj:
            items = obj["results"]
            if not isinstance(items, list):
                raise ValueError("results is not an array")
            native_json = True
        elif isinstance(obj, list):
            items = obj
            native_json = True
        else:
            raise ValueError("unexpected top-level JSON shape")
        raw_entries = list(items)
        raw_items = [item for item in items if isinstance(item, dict) and item.get("key")]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        fallback_used = True
        parse_errors.append(str(exc))
        # Try a whole-document repair first — it recovers cases the line-based parser
        # below can't, e.g. pretty-printed JSON missing only the final closing brace,
        # where no single line is a complete JSON object. On success, feed the
        # recovered items through the same raw_entries/raw_items shape the native-JSON
        # path uses, so the field validation and key-order checks below run unchanged.
        doc_items = _try_repair_document(stripped)
        if doc_items is not None:
            repair_used = True
            raw_entries = list(doc_items)
            raw_items = [item for item in doc_items if isinstance(item, dict) and item.get("key")]
        else:
            for line in stripped.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as line_exc:
                    item = _try_repair(line)
                    if item is not None:
                        repair_used = True
                    else:
                        parse_errors.append(str(line_exc))
                if isinstance(item, dict):
                    raw_entries.append(item)
                    if item.get("key"):
                        raw_items.append(item)

    # The tolerant extraction API intentionally supplies defaults for old
    # callers.  Benchmark validity is stricter: every emitted result must
    # explicitly carry the complete wire contract and valid field types.
    required_fields = ("key", "value", "cited_text", "comment", "confidence", "status")
    schema_violations: list[dict] = []
    missing_fields: list[dict] = []
    allowed_status = {"ok", "absent", "unmappable"}
    for ordinal, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            schema_violations.append({"ordinal": ordinal, "key": None, "fields": ["entry must be an object"]})
            continue
        absent = [field for field in required_fields if field not in item]
        if absent:
            missing_fields.append({"ordinal": ordinal, "key": item.get("key"), "fields": absent})
        invalid: list[str] = []
        if "key" in item and (not isinstance(item["key"], str) or not item["key"]):
            invalid.append("key")
        if "value" in item:
            value = item["value"]
            if not (value is None or isinstance(value, (str, int, float, bool)) or
                    (isinstance(value, list) and all(isinstance(v, str) for v in value))):
                invalid.append("value")
        # null and "" are equivalent on the wire for these two fields — _extract_result
        # normalizes both to "" (via _normalize_cited_text / `obj.get("comment") or ""`),
        # so a model that writes null here is not a schema violation.
        if "cited_text" in item and not (item["cited_text"] is None or isinstance(item["cited_text"], str) or
                                           (isinstance(item["cited_text"], list) and all(isinstance(v, str) for v in item["cited_text"]))):
            invalid.append("cited_text")
        if "comment" in item and not (item["comment"] is None or isinstance(item["comment"], str)):
            invalid.append("comment")
        if "confidence" in item and not (item["confidence"] is None or (isinstance(item["confidence"], int) and not isinstance(item["confidence"], bool))):
            invalid.append("confidence")
        if "status" in item and item["status"] not in allowed_status:
            invalid.append("status")
        if invalid:
            schema_violations.append({"ordinal": ordinal, "key": item.get("key"), "fields": invalid})

    emitted = [str(item["key"]) for item in raw_items]
    seen: set[str] = set()
    duplicates: list[str] = []
    for key in emitted:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    unexpected = list(dict.fromkeys(key for key in emitted if key not in expected_set))
    missing = [key for key in expected if key not in seen]

    # Keep the last duplicate's legacy value, matching parse_structured_output.
    by_key: dict[str, dict] = {}
    for item in raw_items:
        key = str(item["key"])
        if key in expected_set:
            answer = _extract_result(item)
            answer["present"] = True
            by_key[key] = answer
    answers: list[dict] = []
    for key in expected:
        if key in by_key:
            answers.append(by_key[key])
        elif annotate_mode:
            answers.append({
                "key": key, "value": None, "cited_text": "", "comment": "",
                "confidence": None, "status": "absent", "present": False,
            })
        else:
            answers.append({
                "key": key, "value": None, "cited_text": "", "comment": "",
                "confidence": None, "parse_error": f"key {key!r} not found in pass-2 output",
                "present": False,
            })
    return {
        "answers": answers,
        "native_json": native_json,
        "fallback_used": fallback_used,
        "repair_used": repair_used,
        "emitted_key_order": emitted,
        "missing_keys": missing,
        "duplicate_keys": duplicates,
        "unexpected_keys": unexpected,
        "parse_errors": parse_errors,
        "missing_fields": missing_fields,
        "schema_violations": schema_violations,
    }
