"""Verify cited_text is a substring of the source document."""

from __future__ import annotations

import re
import unicodedata
from typing import Union

from fuzzysearch import find_near_matches


def _normalize(text: str) -> str:
    """Collapse whitespace and normalise unicode for fuzzy span matching."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _strip_outer_quotes(text: str) -> str:
    """Strip common wrapping quote characters from both ends."""
    return text.strip('"“”‘’`')


def _fuzzy_find(pattern: str, text: str, max_error_rate: float) -> list:
    """Return all near-matches of pattern in text within max_error_rate Levenshtein distance."""
    if not pattern or not text:
        return []
    max_l_dist = max(1, int(len(pattern) * max_error_rate))
    return find_near_matches(pattern, text, max_l_dist=max_l_dist)


def _fuzzy_in(pattern: str, text: str, max_error_rate: float) -> bool:
    return bool(_fuzzy_find(pattern, text, max_error_rate))


def _verify_single(
    segment: str,
    norm_source: str,
    max_error_rate: float,
    max_ellipsis_gap: int,
) -> bool:
    """Verify one normalized segment against the source, with ellipsis fallback."""
    if _fuzzy_in(segment, norm_source, max_error_rate):
        return True
    parts = re.split(r"\.{2,}|…", segment)
    if len(parts) >= 2:
        return _verify_ellipsis_parts(parts, norm_source, max_error_rate, max_ellipsis_gap)
    return False


def _verify_ellipsis_parts(
    parts: list[str],
    norm_source: str,
    max_error_rate: float,
    max_ellipsis_gap: int = 600,
) -> bool:
    """Verify ellipsis-split parts appear in source in order and within max_ellipsis_gap of each other.

    Anchor strategy: match the longest part first (most unique), then require all
    left-of-anchor parts to appear in the window before the anchor match, and all
    right-of-anchor parts to appear in the window after it.
    """
    normed = [_normalize(_strip_outer_quotes(p)) for p in parts]
    normed = [p for p in normed if p]
    if not normed:
        return False

    # Total length guard: the combined citation must be substantial enough to be meaningful
    if sum(len(p) for p in normed) < 40:
        return False

    if len(normed) == 1:
        return _fuzzy_in(normed[0], norm_source, max_error_rate)

    # Anchor = longest part (highest chance of being unique in the source)
    anchor_idx = max(range(len(normed)), key=lambda i: len(normed[i]))
    anchor = normed[anchor_idx]
    left_parts = normed[:anchor_idx]
    right_parts = normed[anchor_idx + 1:]

    anchor_matches = _fuzzy_find(anchor, norm_source, max_error_rate)
    if not anchor_matches:
        return False

    for m in anchor_matches:
        # Left parts must appear before this anchor match.
        # Extend window by the part's own length so a part that starts right at
        # the gap boundary still fits fully inside the search window.
        left_ok = True
        for lp in left_parts:
            window = norm_source[max(0, m.start - max_ellipsis_gap - len(lp)):m.start]
            if not _fuzzy_in(lp, window, max_error_rate):
                left_ok = False
                break
        if not left_ok:
            continue

        # Right parts must appear after this anchor match (same padding logic).
        right_ok = True
        for rp in right_parts:
            window = norm_source[m.end:m.end + max_ellipsis_gap + len(rp)]
            if not _fuzzy_in(rp, window, max_error_rate):
                right_ok = False
                break
        if right_ok:
            return True

    return False


_SENTINELS = {"", "not found", "n/a"}


def verify_citation(
    cited_text: Union[str, list],
    source: str,
    max_error_rate: float = 0.05,
    max_ellipsis_gap: int = 600,
) -> dict:
    """Return {ok: bool, note: str}."""
    # 1. Empty / sentinel check
    if not cited_text:
        return {"ok": True, "note": "no citation provided"}
    if isinstance(cited_text, str) and cited_text.strip() in _SENTINELS:
        return {"ok": True, "note": "no citation provided"}

    norm_source = _normalize(source)

    # 2. List form: each element is an independent citation, all must match.
    # Elements shorter than 20 chars are skipped (structural markers like "=== Abstract ===").
    # Each element falls back to ellipsis splitting if direct fuzzy fails.
    if isinstance(cited_text, list):
        segments = [_normalize(_strip_outer_quotes(s)) for s in cited_text if s and s.strip()]
        segments = [s for s in segments if len(s) >= 20]
        if not segments:
            return {"ok": True, "note": "no citation provided"}
        if all(_verify_single(s, norm_source, max_error_rate, max_ellipsis_gap) for s in segments):
            return {"ok": True, "note": f"matched {len(segments)} listed segment(s)"}
        return {"ok": False, "note": "one or more listed cited segments not found in source"}

    # From here cited_text is a str
    norm_cited = _normalize(cited_text)

    # 3. Direct fuzzy match — handles outer-quote artifacts (1–2 edit ops ≤ 5% for ≥40-char citations)
    if _fuzzy_in(norm_cited, norm_source, max_error_rate):
        return {"ok": True, "note": ""}

    # 4. Newline-split independent segments (old-style multi-citation strings like "A"\n"B")
    lines = [_normalize(_strip_outer_quotes(ln)) for ln in cited_text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if len(ln) >= 40]
    if len(lines) >= 2 and all(_fuzzy_in(ln, norm_source, max_error_rate) for ln in lines):
        return {"ok": True, "note": f"matched {len(lines)} newline-split segment(s)"}

    # 5. Ellipsis-split ordered segments: "A ... B" or "A … B"
    raw_parts = re.split(r"\.{2,}|…", cited_text)
    if len(raw_parts) >= 2 and _verify_ellipsis_parts(
        raw_parts, norm_source, max_error_rate, max_ellipsis_gap
    ):
        return {"ok": True, "note": f"matched {len(raw_parts)} ellipsis part(s) in order"}

    return {
        "ok": False,
        "note": f"cited span not found in source (first 80 chars): {cited_text[:80]!r}",
    }
