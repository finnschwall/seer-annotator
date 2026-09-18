"""Canonical citation-object builder, shared by every engine call site.

The LLM may cite zero, one, or several verbatim supporting quotes for an
answer. Historically these were flattened with "\\n\\n".join(...) into the
legacy `cited_text` string right before posting to SEER, which destroyed the
multi-citation structure and made per-quote verification impossible to see
downstream. This module builds the structured replacement: a list of citation
objects, one per quote, each carrying its own verification result.

Each object records the quote and what verification made of it:

    {
        "text":         "verbatim quote string",   # required, non-empty str
        "verified":     True,                         # True | False | None
        "reason":       "ok",                         # one of verify.REASONS
        "verifier":     2,                            # verify.VERIFIER_VERSION
    }

**Where the quote is, is not filled in here.**  SEER attaches `block_ids`,
`page_idx`, `section_path`, `section_role` and `locator` on its own side, in
`papers/citation_locate_service.py`, called from the one function every answer
write passes through.  It has to be there: resolving a quote to a block needs the
parsed document and its content-anchored ids, and this library is only ever handed
the rendered text.  Doing it here would mean shipping the block table over the wire
to compute something the receiver can compute exactly.

`reason` and `verifier` live on the citation rather than on the answer because
both are facts about *this quote's verification*: the reason is what the check
found, and the version is which code found it.  Run-level statistics on the SEER
side are read off them, and a rate is only comparable with another run's if both
were produced by the same verifier.

`block_ids` is always a list (never a bare id) because a single quote can
span more than one OCR block (e.g. a sentence split across a page break or a
table cell boundary) — that must not require a schema change when Phase 2
lands. The LLM itself never sees or emits block ids; block resolution is a
deterministic post-processing step run against the OCR output, not something
asked of the model.
"""

from __future__ import annotations

from typing import Callable, Union

from .verify import VERIFIER_VERSION

# An optional hook that maps a quote to location keys merged onto its citation.
# Nothing passes one: SEER places quotes on its own side instead (see the module
# docstring), which is where the parsed document lives. Kept because it costs
# nothing and a local caller with its own block table could still use it.
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
            quotes in cited_text. Each contributes its `ok` flag and its
            `reason`.
        block_matcher: Optional; nothing in the pipeline passes one. SEER
            places quotes itself after the answer arrives. A caller holding its
            own block table can pass a callable mapping a quote's text to
            location keys, which are merged onto each citation object.

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
            "reason": verify.get("reason"),
            "verifier": VERIFIER_VERSION,
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
