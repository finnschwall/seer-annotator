"""Verify cited_text is a substring of the source document."""

from __future__ import annotations

import re
import unicodedata
from typing import Union

from fuzzysearch import find_near_matches

# --- Maths markup -----------------------------------------------------------
#
# MinerU keeps maths as LaTeX, spacing and all: `h _ { r }`,
# `\mathrm { s o f t m a x }`, `$n = 2 6 2 1 4 4$`.  A model quoting the same
# sentence writes `h_r`, `softmax`, `n = 262144`.  Every brace, dollar and stray
# space is an edit against a 5% Levenshtein budget, so a quote that is otherwise
# verbatim gets rejected.  Measured on a labelled sample, this was the only real
# defect among flagged quotes: 8 of 9 contiguous-but-flagged quotes were LaTeX.
#
# This belongs here and not in the rendered source.  SEER's
# `papers/fulltext/markdown.py` measured tidying the source itself and found it a
# wash — a model reading letter-spaced maths sometimes copies it letter-spaced,
# and then the tidied source no longer contains what the model wrote.
# `_normalize` runs over the quote *and* the source, so tidying here is symmetric
# and that failure mode cannot happen: whichever spelling the model copied, both
# sides end up the same.
#
# Deliberately narrow.  Every edit the 5% budget allows is an edit an invented
# quote can also spend, so nothing is rewritten outside an explicit maths span —
# a `$...$` group, a `\command{...}`, or a `_{...}`/`^{...}` script.  Ordinary
# prose, including its punctuation, is left exactly as it was.
#
# And it is a **second chance, not a replacement**: a quote is matched plainly
# first, and only a miss is retried with both sides tidied.  Tidying deletes
# characters, which shrinks the 5% budget along with the string — measured on
# 85k real quotes, doing it unconditionally cost 52 heavily-mathematical quotes
# that had been passing on the slack a long LaTeX string buys.  As a second pass
# it cannot cost anything: whatever matched before still matches in pass one.

#: Cheap pre-check: no span can exist without one of these characters.
_HAS_MATH_RE = re.compile(r'[$\\{]')

#: An explicit maths span.  Only text inside one of these is rewritten.
_MATH_SPAN_RE = re.compile(
    r"""
      \$\$.{0,400}?\$\$                      # display maths
    | \$[^$\n]{1,200}?\$                     # inline maths
    | \\[a-zA-Z]+\s*\{[^{}]{0,200}\}         # \mathrm { s o f t m a x }
    | \\[a-zA-Z]+                            # bare command: \alpha, \times
    | \s*[_^]\s*\{[^{}]{0,200}\}             # h _ { r }  (leading space included,
    """,                                     # so the whole script collapses)
    re.VERBOSE | re.DOTALL,
)

#: Letter-spacing, as the `pipeline` MinerU backend writes it: single
#: alphanumerics separated by single spaces.  At least four in a row — three is
#: as likely to be separate symbols (`x y z`) as a spaced-out word.
_SPACED_RUN_RE = re.compile(r'\b(?:\w ){3,}\w\b')

_MATH_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_SCRIPT_SPACE_RE = re.compile(r'\s*([_^])\s*')


def _tidy_math(span: str) -> str:
    """Reduce one maths span to the spelling a model would quote it as."""
    span = _SPACED_RUN_RE.sub(lambda m: m.group(0).replace(' ', ''), span)
    # A command becomes a space, not nothing: `a\,b` must not fuse into `ab`.
    span = _MATH_CMD_RE.sub(' ', span)
    span = span.replace('$', '')
    span = span.replace('{', '').replace('}', '')
    return _SCRIPT_SPACE_RE.sub(r'\1', span)


def _normalize_math(text: str) -> str:
    if not _HAS_MATH_RE.search(text):
        return text
    return _MATH_SPAN_RE.sub(lambda m: _tidy_math(m.group(0)), text)


def _normalize(text: str, math: bool = False) -> str:
    """Collapse whitespace and normalise unicode for fuzzy span matching.

    `math=True` additionally reduces maths spans — used only for the retry pass.
    """
    text = unicodedata.normalize("NFKC", text)
    if math:
        text = _normalize_math(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


#: An ellipsis between two halves of a quote, however the model spelled it:
#: `...`, `…`, spaced dots, and any of those wrapped in brackets.  The brackets
#: are consumed here, because a `[` left stranded on the end of a part is an
#: edit against the same 5% budget as everything else.
_ELLIPSIS_RE = re.compile(r'\s*[\[\(]?\s*(?:\.\s*\.(?:\s*\.)*|…)\s*[\]\)]?\s*')

#: Bumped whenever a change here can move a stored flag.  A run's numbers are only
#: comparable with another run's if both were verified by the same code, and this is
#: the only thing that says so — see `LLMAnswer.citations_verifier` on the SEER side.
VERIFIER_VERSION = 2

#: Why a quote did or did not match.  A closed set: an unknown value is a bug, not a
#: new category, because run-level statistics are read off these and a typo would
#: quietly become a sixth kind of failure.
#:
#: The failure half is the whole point.  "Invented" and "assembled out of real text in
#: the wrong order" are different facts about a model and both used to render as one
#: unexplained flag.  On a hand-checked sample of 142 flagged quotes these five
#: accounted for every one.
REASONS = (
    'ok',             # found in the paper, contiguous
    'ok_ellipsis',    # found as ordered parts either side of an ellipsis
    'absent',         # none of it is in the paper
    'partial',        # some parts are, some are not
    'reordered',      # every part is there, but not in the order quoted
    'too_far',        # every part is there, in order, but too far apart to be one quote
    'skipped_short',  # too short to be worth checking
)

#: (short label, one sentence of cause).  The only place a reason becomes text —
#: same rule as SEER's `experiments/failure_explain.py`.  Deliberately says what was
#: observed and not what it implies: a quote can be verbatim and still not support
#: the answer, and this check cannot tell.
REASON_LABELS = {
    'ok': ('Verified', 'Found in the paper.'),
    'ok_ellipsis': ('Verified', 'Found as ordered parts either side of an ellipsis.'),
    'absent': ('Not found', 'None of this text is in the paper.'),
    'partial': ('Partly matched', 'Some of this text is in the paper and some is not — a paraphrase or a splice.'),
    'reordered': ('Out of order', 'Every part is in the paper, but not in the order quoted.'),
    'too_far': ('Stitched', 'Every part is in the paper and in order, but too far apart to be one quote.'),
    'skipped_short': ('Not checked', 'Too short to check.'),
}

#: Target piece size for classifying a miss.  60 characters is ~3 edits of slack at
#: the 5% budget — enough for a piece to match on its own words rather than by
#: accident.  20 is the floor, the same length below which a whole quote is not
#: checked at all.
_CHUNK_CHARS = 60
_MIN_PIECE = 20


_EDGE_OPEN = {'[': ']', '(': ')', '{': '}'}
_EDGE_CLOSE = {v: k for k, v in _EDGE_OPEN.items()}
_QUOTE_CHARS = '"“”‘’`'


def _strip_outer_quotes(text: str) -> str:
    """Strip common wrapping quote characters from both ends."""
    return text.strip(_QUOTE_CHARS)


def _strip_edges(text: str) -> str:
    """Strip wrapping quotes, and brackets left unbalanced by the ellipsis split.

    A bracket is only removed when its partner is missing from the part — so
    `(n = 5)` at the end of a part keeps both of its own brackets, and only the
    debris of `(A ... B)` is cleaned up.
    """
    text = _strip_outer_quotes(text.strip()).strip()
    changed = True
    while changed and text:
        changed = False
        if text[0] in _EDGE_OPEN and _EDGE_OPEN[text[0]] not in text:
            text, changed = text[1:].strip(), True
        if text and text[-1] in _EDGE_CLOSE and _EDGE_CLOSE[text[-1]] not in text:
            text, changed = text[:-1].strip(), True
    return _strip_outer_quotes(text).strip()


def _fuzzy_find(pattern: str, text: str, max_error_rate: float) -> list:
    """Return all near-matches of pattern in text within max_error_rate Levenshtein distance."""
    if not pattern or not text:
        return []
    max_l_dist = max(1, int(len(pattern) * max_error_rate))
    return find_near_matches(pattern, text, max_l_dist=max_l_dist)


def _fuzzy_in(pattern: str, text: str, max_error_rate: float) -> bool:
    return bool(_fuzzy_find(pattern, text, max_error_rate))


# --- Public matching primitives ---------------------------------------------
#
# SEER's `papers/fulltext/locate.py` asks a different question about the same
# quote -- not *whether* it is in the paper but *where* -- and it has to reach
# that answer by matching text exactly the way this module does.  Any divergence
# shows up as a quote this module verified and the locator cannot find, which
# reads as a missing feature rather than the disagreement it is.
#
# So these are the same objects, not reimplementations, exported under names that
# are safe to depend on.  Nothing here changes what verification does, and
# VERIFIER_VERSION is deliberately not bumped.

normalize = _normalize
strip_edges = _strip_edges
fuzzy_find = _fuzzy_find
split_ellipsis = _ELLIPSIS_RE.split


def _verify_single(
    segment: str,
    norm_source: str,
    max_error_rate: float,
    max_ellipsis_gap: int,
    math: bool = False,
) -> str:
    """Return how one normalized segment matched — `''` if it did not.

    A reason rather than a bool, because `ok` and `ok_ellipsis` are different facts
    about a quote and the caller has no other way to tell them apart.  Every caller
    tests it for truth, and only the empty string is falsy.
    """
    if _fuzzy_in(segment, norm_source, max_error_rate):
        return 'ok'
    parts = _ELLIPSIS_RE.split(segment)
    if len(parts) >= 2 and _verify_ellipsis_parts(
        parts, norm_source, max_error_rate, max_ellipsis_gap, math=math
    ):
        return 'ok_ellipsis'
    return ''


def _chain_left(parts: list[str], norm_source: str, boundary: int,
                max_error_rate: float, gap: int) -> bool:
    """Match `parts` right-to-left, each within `gap` of the one to its right.

    `boundary` starts at the anchor's start and moves to each match's start, so a
    three-part quote is judged on the gap between *neighbours*.  Measuring every
    part from the anchor instead rejects `A ... B ... C` whenever A and C are more
    than `gap` apart, however tight each individual gap is.
    """
    for part in reversed(parts):
        window_start = max(0, boundary - gap - len(part))
        matches = _fuzzy_find(part, norm_source[window_start:boundary], max_error_rate)
        if not matches:
            return False
        # Rightmost match: the one closest to the part it must sit beside.
        boundary = window_start + max(m.start for m in matches)
    return True


def _chain_right(parts: list[str], norm_source: str, boundary: int,
                 max_error_rate: float, gap: int) -> bool:
    """Match `parts` left-to-right, each within `gap` of the one to its left."""
    for part in parts:
        window_end = min(len(norm_source), boundary + gap + len(part))
        matches = _fuzzy_find(part, norm_source[boundary:window_end], max_error_rate)
        if not matches:
            return False
        boundary += min(m.end for m in matches)
    return True


def _verify_ellipsis_parts(
    parts: list[str],
    norm_source: str,
    max_error_rate: float,
    max_ellipsis_gap: int = 600,
    math: bool = False,
) -> bool:
    """Verify ellipsis-split parts appear in source in order and within max_ellipsis_gap of each other.

    Anchor strategy: match the longest part first (most unique), then chain
    outwards from it — each remaining part must appear within max_ellipsis_gap of
    its own neighbour, not of the anchor.
    """
    normed = [_normalize(_strip_edges(p), math=math) for p in parts]
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

    for m in _fuzzy_find(anchor, norm_source, max_error_rate):
        if not _chain_left(left_parts, norm_source, m.start, max_error_rate, max_ellipsis_gap):
            continue
        if _chain_right(right_parts, norm_source, m.end, max_error_rate, max_ellipsis_gap):
            return True

    return False


_SENTINELS = {"", "not found", "n/a"}


class _Source:
    """The source document in both spellings; the maths pass is built on demand.

    Tidying maths costs a scan of the whole document, and most answers never
    reach the retry, so it is only paid for when a quote actually misses.
    """

    def __init__(self, source: str):
        self._raw = source
        self.plain = _normalize(source)
        self._math = None

    @property
    def math(self) -> str:
        if self._math is None:
            self._math = _normalize(self._raw, math=True)
        return self._math


def _chunks(segment: str) -> list[str]:
    """Split a quote into roughly equal pieces, on word boundaries.

    Always at least two pieces once the quote is long enough to hold two: with one
    piece there is nothing to compare, so a half-invented quote would be indexed as
    wholly absent.
    """
    words = segment.split()
    if not words:
        return []
    n = max(1, round(len(segment) / _CHUNK_CHARS))
    if len(segment) >= 2 * _MIN_PIECE:
        n = max(2, n)
    target = len(segment) / n

    out, cur = [], ''
    for word in words:
        cur = f'{cur} {word}'.strip()
        if len(cur) >= target and len(out) < n - 1:
            out.append(cur)
            cur = ''
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= _MIN_PIECE]


def _classify_miss(segment: str, src: "_Source", max_error_rate: float, max_ellipsis_gap: int) -> str:
    """Say *why* a quote did not match, as one of `REASONS`.

    Runs only on a miss — about 4% of quotes — so the cost of a second pass over the
    document is not paid by the quotes that matched.

    The pieces are the quote's own ellipsis parts where it has them, and fixed-size
    chunks where it does not, so a stitched quote is judged on the parts the model
    actually wrote.  Matching is done against the maths-tidied source, because the
    question here is whether the *words* are in the paper at all.
    """
    parts = [_normalize(_strip_edges(p), math=True) for p in _ELLIPSIS_RE.split(segment)]
    parts = [p for p in parts if len(p) >= _MIN_PIECE]
    if len(parts) < 2:
        parts = _chunks(_normalize(segment, math=True))
    if not parts:
        return 'absent'

    found = [_fuzzy_find(p, src.math, max_error_rate) for p in parts]
    n_found = sum(1 for f in found if f)
    if n_found == 0:
        return 'absent'
    if n_found < len(parts):
        return 'partial'

    # Every piece is somewhere in the paper. Walk them left to right, always taking
    # the earliest match that still lies after the previous one: if no such match
    # exists the quote reorders the paper's own text.
    cursor, spans = 0, []
    for matches in found:
        ahead = [m for m in matches if m.start >= cursor]
        if not ahead:
            return 'reordered'
        best = min(ahead, key=lambda m: m.start)
        spans.append((best.start, best.end))
        cursor = best.end

    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    if gaps and max(gaps) > max_ellipsis_gap:
        return 'too_far'
    # In order and close together, yet the whole thing still did not match: the joins
    # between the pieces are not what the paper says.
    return 'partial'


def _verify_segment(raw: str, src: "_Source", max_error_rate: float,
                    max_ellipsis_gap: int, classify: bool = True) -> tuple[bool, str]:
    """Match one quote, plainly first and then with maths markup tidied.

    Returns `(matched, reason)`; the reason is one of `REASONS` either way.

    `classify=False` skips naming *why* a miss missed, for the legacy whole-blob
    caller that has nowhere to put the answer — every engine call site runs both
    entry points over the same quotes, so classifying twice is a scan of the
    document nobody reads.
    """
    segment = _normalize(_strip_edges(raw))
    reason = _verify_single(segment, src.plain, max_error_rate, max_ellipsis_gap)
    if reason:
        return True, reason

    math_segment = _normalize(_strip_edges(raw), math=True)
    if math_segment != segment or src.math != src.plain:
        reason = _verify_single(math_segment, src.math, max_error_rate, max_ellipsis_gap, math=True)
        if reason:
            return True, reason

    if not classify:
        return False, ''
    return False, _classify_miss(segment, src, max_error_rate, max_ellipsis_gap)


def _verify_whole(cited_text: str, src: "_Source", max_error_rate: float,
                  max_ellipsis_gap: int, math: bool) -> "dict | None":
    """One pass of the whole-string paths: direct, newline-split, ellipsis-split."""
    norm_source = src.math if math else src.plain

    # Direct fuzzy match — handles outer-quote artifacts (1–2 edit ops ≤ 5% for ≥40-char citations)
    if _fuzzy_in(_normalize(cited_text, math=math), norm_source, max_error_rate):
        return {"ok": True, "note": ""}

    # Newline-split independent segments (old-style multi-citation strings like "A"\n"B")
    lines = [_normalize(_strip_edges(ln), math=math) for ln in cited_text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if len(ln) >= 40]
    if len(lines) >= 2 and all(_fuzzy_in(ln, norm_source, max_error_rate) for ln in lines):
        return {"ok": True, "note": f"matched {len(lines)} newline-split segment(s)"}

    # Ellipsis-split ordered segments: "A ... B" or "A … B"
    raw_parts = _ELLIPSIS_RE.split(cited_text)
    if len(raw_parts) >= 2 and _verify_ellipsis_parts(
        raw_parts, norm_source, max_error_rate, max_ellipsis_gap, math=math
    ):
        return {"ok": True, "note": f"matched {len(raw_parts)} ellipsis part(s) in order"}

    return None


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

    src = _Source(source)

    # 2. List form: each element is an independent citation, all must match.
    # Elements shorter than 20 chars are skipped (structural markers like "=== Abstract ===").
    # Each element falls back to ellipsis splitting if direct fuzzy fails.
    if isinstance(cited_text, list):
        # str(s): elements are normally strings, but a badly repaired JSON item can
        # put a nested list or dict here. Coerce rather than crash — same tolerance
        # as verify_citations() below. The malformed item is flagged separately by
        # annotate.parse.cited_text_violation.
        raws = [str(s) for s in cited_text if s and str(s).strip()]
        raws = [r for r in raws if len(_normalize(_strip_edges(r))) >= 20]
        if not raws:
            return {"ok": True, "note": "no citation provided"}
        if all(_verify_segment(r, src, max_error_rate, max_ellipsis_gap, classify=False)[0]
               for r in raws):
            return {"ok": True, "note": f"matched {len(raws)} listed segment(s)"}
        return {"ok": False, "note": "one or more listed cited segments not found in source"}

    # From here cited_text should be a str; coerce anything else (a dict, a number)
    # so this back-compat entry point can never raise on an unexpected shape.
    if not isinstance(cited_text, str):
        cited_text = str(cited_text)

    # 3. The whole-string paths, plainly and then with maths markup tidied.
    for math in (False, True):
        hit = _verify_whole(cited_text, src, max_error_rate, max_ellipsis_gap, math)
        if hit:
            return hit

    return {
        "ok": False,
        "note": f"cited span not found in source (first 80 chars): {cited_text[:80]!r}",
    }


def verify_citations(
    cited_text: Union[str, list, None],
    source: str,
    max_error_rate: float = 0.05,
    max_ellipsis_gap: int = 600,
) -> list[dict]:
    """Return one {ok, reason, note} verification result per quote, preserving order.

    Unlike verify_citation() (which collapses a whole cited_text into a single
    ok/note pair for back-compat), this gives per-quote verification so each
    citation object on the wire can carry its own `verified` flag. Reuses the
    same normalization/fuzzy-match/ellipsis-fallback helpers as verify_citation.

    A str input is wrapped as a single-element list so callers always get a
    list back, regardless of whether the LLM emitted one quote or several.

    `reason` is one of `REASONS` and is the only thing that says *why* a quote
    failed; the flag alone cannot tell an invented quote from one assembled out of
    real text in the wrong order, and those are different facts about a model.
    """
    if not cited_text:
        return []
    if isinstance(cited_text, str) and cited_text.strip() in _SENTINELS:
        return []

    src = _Source(source)
    quotes = cited_text if isinstance(cited_text, list) else [cited_text]

    results = []
    for raw in quotes:
        if not raw or not str(raw).strip():
            results.append({"ok": True, "note": "no citation provided"})
            continue

        # Structural markers / very short fragments are skipped, same threshold
        # as the list branch of verify_citation.
        if len(_normalize(_strip_edges(str(raw)))) < 20:
            results.append({
                "ok": True, "note": "skipped (too short to verify)", "reason": "skipped_short",
            })
            continue

        ok, reason = _verify_segment(str(raw), src, max_error_rate, max_ellipsis_gap)
        results.append(
            {
                "ok": ok,
                "reason": reason,
                "note": "" if ok else f"cited span not found in source (first 80 chars): {str(raw)[:80]!r}",
            }
        )

    return results
