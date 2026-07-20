"""Canonical citation-object builder, shared by every engine call site.

The LLM may cite zero, one, or several verbatim supporting quotes for an
answer. Historically these were flattened with "\\n\\n".join(...) into the
legacy `cited_text` string right before posting to SEER, which destroyed the
multi-citation structure and made per-quote verification impossible to see
downstream. This module builds the structured replacement: a list of citation
objects, one per quote, each carrying its own verification result.

Phase 1 (now) fills only `text` and `verified` on each citation object.
Everything else in the schema below is a deliberate seam for Phase 2, which
will resolve each quote to the OCR block(s) it actually appears in:

    {
        "text":         "verbatim quote string",   # required, non-empty str
        "verified":     True,                         # True | False | None
        "block_ids":    ["a1b2...", ...],            # Phase 2 (omit/[] now)
        "page_idx":     3,                            # Phase 2
        "section_path": "Methods > Participants",     # Phase 2
        "char_offset":  [1204, 1337],                  # Phase 2
    }

`block_ids` is always a list (never a bare id) because a single quote can
span more than one OCR block (e.g. a sentence split across a page break or a
table cell boundary) — that must not require a schema change when Phase 2
lands. The LLM itself never sees or emits block ids; block resolution is a
deterministic post-processing step run against the OCR output, not something
asked of the model.
"""

from __future__ import annotations

from typing import Callable, Union

# Phase-2 seam: a callable (quote_text: str) -> dict that resolves a verbatim
# quote to its block_ids/page_idx/section_path/char_offset within the source
# document. Not implemented yet — build_citations() accepts it as an optional
# argument so call sites don't need to change again when it lands.
BlockMatcher = Callable[[str], dict]


def build_citations(
    cited_text: Union[str, list, None],
    verify_results: list[dict],
    block_matcher: "BlockMatcher | None" = None,
) -> list[dict]:
    """Build the list of wire-ready citation objects for one answer.

    Args:
        cited_text: The LLM's quote(s) as already normalized by
            annotate.parse._normalize_cited_text — a str, a list[str], or
            None/"" when the LLM signalled no verbatim quote is available
            (the `[NO DIRECT QUOTE]` sentinel normalizes to None upstream).
        verify_results: Per-quote verification results from
            annotate.verify.verify_citations(), in the same order as the
            quotes in cited_text.
        block_matcher: Phase-2 seam, unused in Phase 1. When the OCR
            block-resolution step is implemented, pass a callable here that
            maps a quote's text to a dict of block_ids/page_idx/section_path/
            char_offset; those keys will be merged onto each citation object.

    Returns:
        A list of citation dicts (see module docstring for the schema).
        Empty list when there is no citation to carry.
    """
    if not cited_text:
        return []
    if isinstance(cited_text, str) and not cited_text.strip():
        return []

    quotes = cited_text if isinstance(cited_text, list) else [cited_text]

    citations = []
    for i, quote in enumerate(quotes):
        if not quote or not str(quote).strip():
            continue
        verify = verify_results[i] if i < len(verify_results) else {}
        citation = {
            "text": str(quote),
            "verified": verify.get("ok"),
        }
        if block_matcher is not None:
            citation.update(block_matcher(citation["text"]))
        citations.append(citation)

    return citations


def rollup_verified(citations: list[dict]) -> str:
    """Collapse per-quote `verified` flags into one status string.

    Mirrors the equivalent rollup on the SEER side, for local-store parity
    (dry-run printing, cost/stat summaries, etc.):
      - "all"  : every citation verified True
      - "any"  : at least one True, but not all
      - "none" : at least one False, no True
      - ""     : no citations, or none were checked (all verified is None)
    """
    if not citations:
        return ""
    flags = [c.get("verified") for c in citations]
    if all(f is True for f in flags):
        return "all"
    if any(f is True for f in flags):
        return "any"
    if any(f is False for f in flags):
        return "none"
    return ""


def citation_count(citations: list[dict]) -> int:
    """Number of citation objects, for local-store parity with the SEER side."""
    return len(citations or [])
