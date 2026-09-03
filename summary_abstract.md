# Pass-2 formatting: what 500 abstracts told us

Pass-2 is the cheap second LLM call in the annotation pipeline. It gets Pass-1's
free-form text and turns it into typed JSON. It is not supposed to think — its
only job is faithful restructuring, so for any given Pass-1 text there is exactly
one correct output. That makes every mistake objectively checkable against Pass-1.

We ran six models over the same 500 abstracts to answer two questions: can a
Pass-2 model invent answers, and which model should we use.

**Short version.** Every model invented at least one answer. The pipeline now
blocks that in code rather than trusting the model, so fabrication is no longer a
reason to pick one model over another. What still separates them is citation
fidelity, dropped answers, speed and cost. Use `gemma4-31b`. Do not use
`gpt-5-nano`.

---

## The data

500 abstracts from source run 36, seed 20260825, four inclusion-criteria
questions each — 2000 answers per model, 3000 executions, zero errors. Pass-1 was
`deepseek-v4` for every case, so all Pass-1 text shares one house style; a second
source run would test robustness to a different one.

531 of the 2000 question-answers (26.6%) have no Pass-1 answer block at all.
Pass-1 routinely answers only some questions — often just the inclusion gate that
decides the paper — so "there is nothing here to format" is a quarter of Pass-2's
job, not an edge case. Those 531 are the surface on which a model can invent an
answer.

Everything below is in `benchmarking/bench500/`: `phase2-500.sqlite` (the raw
run), `analysis.txt` and `report.csv` (the numbers), `disagreements.json` (every
disagreement with the Pass-1 text that settles it).

---

## Finding 1: every model invents answers

A **fabrication** is Pass-2 reporting a usable status for a question Pass-1 never
answered — an answer conjured from nothing.

| model | fabrications / 531 | rate | carried a real value |
|---|---|---|---|
| gemma4-31b | 1 | 0.19% | 1 |
| qwen3.5-397b | 1 | 0.19% | 1 |
| deepseek-v4-pro | 1 | 0.19% | 1 |
| gpt-5-mini | 4 | 0.75% | 4 |
| gpt-5-nano | 9 | 1.69% | 9 |
| deepseek-v4-flash-nothink | 13 | 2.45% | 13 |

Every fabrication carried a non-null value, so every one of them would have been
stored as a real annotation. None were empty placeholders.

Fabrication does not track model size or price. `gpt-5-mini` invents four times as
often as a 31B open model. This is a formatting-discipline habit, not a capability
gap, which is exactly why enforcing it in code beats shopping for a better model.

Two situations trigger it.

**An answer block with no key.** In one case Pass-1 wrote the template
placeholder `--- ANSWER: <key> ---` literally instead of substituting a question
key. All six models then guessed which question that orphaned block belonged to.
`gpt-5-mini` and `deepseek-v4-flash` assigned it to all four questions at
confidence 20, inventing mutually contradictory answers from one block that
answers none of them. This single case is the *only* fabrication committed by
`gemma4-31b`, `qwen3.5-397b` and `deepseek-v4-pro`.

**Pass-1 answering only the decisive gate.** `deepseek-v4-flash`'s other 12
fabrications come from three cases where Pass-1 wrote one block and Pass-2 filled
in the other three questions from it.

### This is not a hypothetical

On the placeholder case, every model's raw output silently **excludes the paper**
on a fabricated `xai_operation` answer. With the rule enforced, nothing excludes
it and four visible errors appear instead. That is the right outcome: an
unattributable answer must not decide a paper's fate.

### The guard, and what it costs

`annotate/scope.py` now demotes any answer to `absent` unless Pass-1 actually
contains a block for that key. It runs before the inclusion-criteria computation,
so a fabricated value cannot move the gate even transiently. It only ever demotes
— code cannot invent a value that is genuinely missing, so the reverse direction
is not fixable this way.

The risk was that a Pass-1 model wobbling on header formatting would lose a good
answer. It does not happen: across 500 cases the count of `Answer:` lines matches
the count of answer blocks in 499, and the exception is the placeholder case where
demoting is correct. Cost is 4 papers needing re-annotation for the worst model
and 1 for the best.

### Known gap in the guard

The check asks whether a block *exists*, not whether it *contains an answer*. One
block in 1469 has a real header whose body is
`(Not answered because the previous inclusion-criteria question excluded the paper: xai_operation = No.)`.
The header is present, so the guard stays silent. `gemma4-31b` is the only model
that reports `ok` there; the others say `absent` or `unmappable`.

Harm is negligible today — all six emitted a null value, so nothing false is
stored, and a null value produces no inclusion-criteria signal. The fix is to also
require an `Answer:` line inside the block. Not yet done.

---

## Finding 2: which model to use

Fabrication is now handled in code, so it should not drive the choice. These
columns should.

| model | text lost from citations | dropped answers | native JSON | out tok | $/1k groups | p50 |
|---|---|---|---|---|---|---|
| **gemma4-31b** | 2 | 0 | 99.6% | 558 | free | 2.6s |
| gpt-5-mini | **0** | 1 | 100% | 1363 | $3.02 | 11.3s |
| qwen3.5-397b | **0** | 0 | 99.8% | 5303 | free | 61.2s |
| deepseek-v4-pro | 1 | 0 | 100% | 1831 | free | 12.2s |
| deepseek-v4-flash-nothink | 2 | 1 | 98.8% | 482 | free | 2.0s |
| gpt-5-nano | 19 | 42 | 100% | 4228 | $1.75 | 31.6s |

**Use `gemma4-31b`.** Fastest of the accurate models, free, 558 output tokens per
group, no dropped answers, and minimum possible fabrication. It loses quoted text
twice in 1441 citations — real evidence loss, not a formatting nit — which is the
one honest mark against it.

**`gpt-5-mini` is the alternative if citation completeness matters most.** It is
the only model with zero citation defects at all. The price is four times gemma's
fabrication rate (now neutralised by the guard), four times the latency, and $3
per thousand groups against free.

**`qwen3.5-397b` is the quality leader on paper** — zero citation loss, zero
dropped answers — but 61 seconds and 5303 output tokens per group make it
impractical at scale. It is the right choice for an evaluation reference.

**Reject `gpt-5-nano`.** It drops 42 answered questions outright, omits 76 keys
concentrated in the tail of the answer list, loses citation text 19 times, and
produced 32 structurally broken groups out of 500. Its JSON always parses — Azure's
schema enforcement keeps the envelope well-formed — but the content collapses
inside it, which is worse than a parse failure because nothing downstream flags
it. Eight responses degenerate into repeated-token loops; the worst spent 86,509
output tokens on four questions. It burns 3622 reasoning tokens per group to
perform worse than a 31B model.

**Reject `deepseek-v4-flash` despite the fix.** Disabling reasoning repaired its
wire behaviour completely: native JSON went from 80.5% to 98.8% and p95 latency
from 326s to 4.1s. It did not repair judgement. It fabricates most of any model
tested (2.45%), fails citation verification 21 times against everyone else's 6,
and in 14 cases sets `status: absent` while still populating `value` — a contract
violation that happens to be inert, because the pipeline keys off status.

---

## Finding 3: four things that distort a naive reading

These cost real time to untangle. Anyone rerunning this benchmark should know them
before trusting a number.

**The reference is not ground truth.** `deepseek-v4-pro` served as the comparison
reference, and 10 of the 69 non-nano disagreements are its own errors — a dropped
quote bullet, backslash-escape leakage, inconsistent quote handling. Corrected
mistake counts: gemma 6→3, qwen 5→3, gpt-5-mini 10→7, flash 48→46. Reference
agreement understates the good models. Where a claim needs to be solid, measure
against Pass-1 instead; it is the only ground truth available, and it is enough.

**The GPT-5 models keep the `Reasoning: ` label.** They copy Pass-1's field label
into the comment; every other model strips it. The text is otherwise identical.
This one convention accounted for 175 of `gpt-5-mini`'s 182 comment mismatches and
dragged its group accuracy from a true 98.80% down to an apparent 87.00%. If a
GPT-5 model is ever used for Pass-2, normalise this prefix in `parse.py`.

**A broken group voids its own correct answers.** 45 of the 174 rows flagged
against `gpt-5-nano` are answers *identical* to the reference, scored as failures
because the group they belonged to had broken JSON. That is a defensible scoring
rule — you cannot trust any answer from a malformed response — but it means nano's
65.80% raw answer accuracy mixes "wrong" with "voided". It has 129 real
disagreements, 128 of them its own fault.

**Quote delimiters are genuinely ambiguous.** Pass-1 writes each quote as
`- "span"`. The prompt says copy verbatim but never says whether the delimiting
quote marks are part of the span. Models split completely: gemma strips them
always (0 of 2806), qwen keeps them almost always (2418 of 2808), the rest sit in
between, and `deepseek-v4-pro` is inconsistent with itself at 45%. The citation
verifier is fuzzy so it does not care, but this depresses every agreement score.
It is a prompt defect, not a model failing, and one sentence in
`build_format_messages` would settle it.

---

## What to fix next

1. **Extend the guard** to require an `Answer:` line inside the block, closing the
   skip-stub gap above.
2. **Say whether quote delimiters belong in `cited_text`** in
   `annotate/prompt.py`. Cheapest available accuracy gain.
3. **Fail loudly on an unsubstituted `<key>` header in Pass-1.** It is one case in
   500, but it is the single defect that makes every model fabricate.
4. **Benchmark a second source run.** All 500 cases share one Pass-1 model's house
   style, so nothing here shows how these models behave on differently formatted
   Pass-1 text.
