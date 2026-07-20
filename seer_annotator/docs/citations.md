# Structured citations

## The problem this replaces

When an LLM supports an answer with more than one verbatim quote, the model
already emits them as a list (`cited_text: ["quote one", "quote two"]`) —
`parse.py` and `verify_citation()` both understood multi-quote lists from the
start. But at the wire boundary, `seer_client.py`'s `_coerce_cited_text()`
flattened any list down with:

```python
"\n\n".join(str(v) for v in value)
```

before POSTing to SEER. That destroyed the structure: SEER only ever saw one
opaque string per answer, with no way to know how many quotes were cited,
which ones verified against the source and which didn't, or where each quote
lives in the document. A single `cited_text_verified` boolean covered the
*whole* joined blob — if one quote out of three failed to verify, the whole
answer's citation showed as unverified with no way to tell which quote was
the problem.

## The new wire contract

Every `LLMAnswer` payload now carries a `citations` field alongside (not
instead of) the legacy `cited_text` / `cited_text_verified` fields:

```jsonc
{
  "...": "...",
  "cited_text": "quote one\n\nquote two",   // legacy, unchanged, still joined
  "cited_text_verified": false,             // legacy, unchanged, collapsed bool
  "citations": [
    {
      "text": "quote one",
      "verified": true
    },
    {
      "text": "quote two",
      "verified": false
    }
  ]
}
```

`citations` is always a list (possibly empty when the LLM cited nothing, or
signalled `[NO DIRECT QUOTE]`). Each element is a citation object:

```jsonc
{
  "text":         "verbatim quote string",   // required, non-empty str
  "verified":     true,                        // required: true | false | null
  "block_ids":    ["a1b2...", ...],           // optional list — Phase 2, omitted/[] for now
  "page_idx":     3,                           // optional int  — Phase 2
  "section_path": "Methods > Participants",    // optional str  — Phase 2
  "char_offset":  [1204, 1337]                 // optional [start, end] — Phase 2
}
```

Both the legacy fields and `citations` are populated from the same
underlying data — nothing is dropped, this is purely additive. Consumers
that only understand `cited_text`/`cited_text_verified` keep working
unchanged; consumers that want per-quote structure can read `citations`.

## Per-quote `verified`

Legacy `cited_text_verified` is one bool for the whole (possibly multi-quote)
citation: if any quote in the joined string failed to fuzzy-match the source,
the whole thing is `false`. That is still computed exactly as before, for
back-compat.

`citations[i].verified` is per-quote: each quote is fuzzy-matched against the
source document independently (same matcher as the legacy path — normalized
whitespace/unicode, Levenshtein-distance fuzzy substring search, with an
ellipsis-split fallback for quotes like `"A ... B"`). Its value is:

- `true` — the quote was found in the source (within the configured error
  rate).
- `false` — the quote was checked and not found.
- `null` — the quote was not checked (e.g. too short — under 20 chars after
  normalization — to verify meaningfully; treated the same as the legacy
  list-branch skip for structural markers like `=== Abstract ===`).

This means a reviewer (human or downstream tooling) can now see *exactly*
which of several cited quotes is questionable, instead of only knowing that
*something* in the citation didn't verify.

## The LLM never sees or emits block ids

`block_ids`, `page_idx`, `section_path`, and `char_offset` are **Phase 2**
fields. They are part of the schema today only as a seam — deliberately left
unpopulated (`build_citations()` never sets them, and the LLM prompts and
`parse.py` schema are untouched by this change). The model's job stays what
it already was: return the verbatim quote text it used to support an answer,
nothing else.

Resolving a quote's text to the OCR block(s) it lives in is a **deterministic
post-processing step**, run after the model has responded, against the
already-OCR'd document. It is not, and will never be, something asked of the
LLM — block ids are an artifact of *our* OCR pipeline's chunking, not
something a language model has (or should need) visibility into.

## Phase 2 end state, including multi-block-spanning quotes

When the block-resolution matcher lands, `build_citations()` will accept a
`block_matcher` callable (the seam already exists in
`annotate/citation.py`) that maps a quote's text to the block(s) it was
found in. The result gets merged onto that quote's citation object:

```jsonc
{
  "text": "as shown in Table 2, mean age was 54.3 years across both arms",
  "verified": true,
  "block_ids": ["blk_0f3a", "blk_0f3b"],
  "page_idx": 5,
  "section_path": "Results > Baseline characteristics",
  "char_offset": [204, 268]
}
```

Critically, `block_ids` is **always a list**, never a bare single id — even
in Phase 1 where it's omitted entirely. A single verbatim quote can span more
than one OCR block: a sentence broken across a page boundary, a quote that
starts in a paragraph and continues into a table cell, or two adjacent OCR
blocks that a chunker split mid-sentence. Modeling `block_ids` as a scalar
from the start would have forced a breaking schema change the day the first
multi-block quote showed up; modeling it as a list from day one means Phase 2
is purely additive, exactly like this Phase 1 change was.
