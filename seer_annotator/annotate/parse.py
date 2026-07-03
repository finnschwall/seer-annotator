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


def _parse_raw_value(raw: str) -> object:
    """Convert a regex-captured primitive token to a Python value."""
    if raw in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[raw]
    try:
        return int(raw)
    except ValueError:
        return float(raw)


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


_NO_DIRECT_QUOTE = "[NO DIRECT QUOTE]"


def _normalize_cited_text(raw: object) -> str | list | None:
    """Return None when the LLM signalled no verbatim quote is available."""
    if raw is None:
        return ""
    if isinstance(raw, list):
        stripped = [s.strip() if isinstance(s, str) else s for s in raw]
        if stripped == [_NO_DIRECT_QUOTE]:
            return None
        return raw
    if isinstance(raw, str) and raw.strip() == _NO_DIRECT_QUOTE:
        return None
    return raw or ""


def _extract_result(obj: dict) -> dict:
    """Pull the standard fields out of a parsed JSON object."""
    return {
        "key": obj.get("key"),
        "value": obj.get("value"),
        "cited_text": _normalize_cited_text(obj.get("cited_text")),
        "comment": obj.get("comment") or "",
        "confidence": obj.get("confidence"),
    }


def _fill_missing(results: dict[str, dict], question_keys: list[str]) -> list[dict]:
    out = []
    for k in question_keys:
        if k in results:
            out.append(results[k])
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


def parse_structured_output(text: str, question_keys: list[str]) -> list[dict]:
    """Parse pass-2 output when response_format=json_object was used.

    Expects {"results": [...]} or a bare [...] as the top-level JSON value.
    Falls back to parse_format_output() if the response is not valid JSON or
    does not match the expected shape (e.g. when drop_params silently removed
    response_format).
    """
    text = _strip_code_fence(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "parse_structured_output: response is not valid JSON — falling back to "
            "line-by-line parser. First 200 chars: %.200r", text
        )
        return parse_format_output(text, question_keys)

    if isinstance(obj, dict) and "results" in obj:
        items = obj["results"]
    elif isinstance(obj, list):
        items = obj
    else:
        logger.warning(
            "parse_structured_output: unexpected JSON shape — falling back to "
            "line-by-line parser. type=%s", type(obj).__name__
        )
        return parse_format_output(text, question_keys)

    results: dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and item.get("key"):
            r = _extract_result(item)
            results[r["key"]] = r

    return _fill_missing(results, question_keys)


def parse_format_output(text: str, question_keys: list[str]) -> list[dict]:
    """Extract one dict per question from pass-2 JSON-lines output (repair/opt-out path).

    Returns list of {key, value, cited_text, comment, confidence} dicts.
    Missing/malformed keys get value=None with a parse_error note.
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

        key = obj.get("key")
        if key:
            r = _extract_result(obj)
            results[key] = r

    return _fill_missing(results, question_keys)
