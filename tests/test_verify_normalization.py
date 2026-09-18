"""How a verbatim quote survives LaTeX markup and a stitched-quote ellipsis.

Two defects, both measured on a labelled sample of real answers:

1. MinerU writes maths as LaTeX with its braces and spacing intact, and a model
   quoting the sentence writes the maths the way a reader would. Every brace and
   stray space was an edit against the 5% budget, so a verbatim quote failed. Of
   the flagged quotes that really were contiguous in the paper, 8 of 9 were this.

2. The ellipsis path split on bare dots only, so `[...]` left a bracket glued to
   the part beside it; and it measured every part's distance from the anchor
   rather than from its own neighbour, so a three-part quote failed whenever its
   two ends were more than one window apart.

The maths rule is narrow on purpose: the 5% budget is what stops an invented
quote matching, so nothing outside an explicit maths span is touched.
"""

from seer_annotator.annotate.citation import build_citations
from seer_annotator.annotate import verify as verify_mod
from seer_annotator.annotate.verify import (
    REASON_LABELS,
    REASONS,
    VERIFIER_VERSION,
    _normalize,
    verify_citation,
    verify_citations,
)


def ok(quote, source):
    return verify_citation(quote, source)["ok"]


# --- Maths markup -----------------------------------------------------------

LATEX_SOURCE = (
    "We train the retriever end to end. The hidden state $h _ { r }$ is passed "
    "through \\mathrm { s o f t m a x } to obtain the attention weights, and the "
    "corpus contains $n = 2 6 2 1 4 4$ documents in total."
)


def test_subscript_quoted_without_braces():
    assert ok("The hidden state $h_r$ is passed through softmax to obtain the attention weights", LATEX_SOURCE)


def test_letter_spaced_command_quoted_as_a_word():
    assert ok("is passed through softmax to obtain the attention weights, and the corpus", LATEX_SOURCE)


def test_letter_spaced_number_quoted_as_a_number():
    assert ok("and the corpus contains n = 262144 documents in total", LATEX_SOURCE)


def test_quote_that_copies_the_source_spelling_still_matches():
    """The reason this is done in the verifier and not in the rendered source.

    A model reading letter-spaced maths sometimes copies it letter-spaced.
    Tidying only the source breaks exactly these quotes; tidying both sides
    cannot, because whichever spelling the model chose, the two ends up equal.
    """
    assert ok("state $h _ { r }$ is passed through \\mathrm { s o f t m a x } to obtain the attention", LATEX_SOURCE)


def test_prose_is_not_rewritten():
    """A quote whose words differ is still a miss — the budget is not widened."""
    assert not ok(
        "The hidden state $h_r$ is passed through a sigmoid to obtain the attention weights",
        LATEX_SOURCE,
    )


def test_spaced_out_prose_outside_maths_is_left_alone():
    """Letter-spacing is only joined inside a maths span, never in running text.

    Asserted on the normalizer rather than on a verdict: over a sentence this
    long the 5% budget absorbs three spaces on its own, so a passing verdict
    would not show which of the two allowed it.
    """
    prose = "The three groups were labelled a b c d in the protocol."
    assert _normalize(prose) == prose.lower()


def test_ordinary_punctuation_is_left_alone():
    """Only maths markup is removed. Brackets, quotes and dashes still count."""
    prose = 'He called it a "third way" (see Fig. 2) - with caveats.'
    assert _normalize(prose) == prose.lower()


# --- Ellipsis handling ------------------------------------------------------

STITCH_SOURCE = (
    "Participants were recruited from three universities over two semesters. "
    + "Filler sentence about the apparatus and the room. " * 6
    + "All analyses were pre-registered before data collection began."
)


def test_bracketed_ellipsis():
    assert ok(
        "Participants were recruited from three universities [...] All analyses were pre-registered",
        STITCH_SOURCE,
    )


def test_spaced_dots_inside_brackets():
    assert ok(
        "Participants were recruited from three universities [. . .] All analyses were pre-registered",
        STITCH_SOURCE,
    )


def test_parenthesised_ellipsis():
    assert ok(
        "Participants were recruited from three universities (...) All analyses were pre-registered",
        STITCH_SOURCE,
    )


THREE_PART_SOURCE = (
    "The first claim concerns sample size and statistical power in the study. "
    + "Padding sentence one that carries no claim at all here. " * 7
    + "The second claim concerns the choice of baseline model used throughout. "
    + "Padding sentence two that carries no claim at all here. " * 7
    + "The third claim concerns the generalisation of these results elsewhere."
)


def test_three_parts_each_within_a_window_of_its_neighbour():
    """Ends more than one window apart, neighbours comfortably inside one."""
    assert ok(
        "The first claim concerns sample size and statistical power ... "
        "The second claim concerns the choice of baseline model ... "
        "The third claim concerns the generalisation of these results",
        THREE_PART_SOURCE,
    )


def test_parts_in_the_wrong_order_still_fail():
    assert not ok(
        "The third claim concerns the generalisation of these results ... "
        "The first claim concerns sample size and statistical power",
        THREE_PART_SOURCE,
    )


def test_parts_further_apart_than_the_window_still_fail():
    far_source = (
        "The first claim concerns sample size and statistical power in the study. "
        + "Padding sentence that carries no claim at all in this paper. " * 40
        + "The third claim concerns the generalisation of these results elsewhere."
    )
    assert not ok(
        "The first claim concerns sample size and statistical power ... "
        "The third claim concerns the generalisation of these results",
        far_source,
    )


def test_invented_quote_still_fails():
    assert not ok(
        "Participants were compensated with course credit and a fixed cash payment.",
        STITCH_SOURCE,
    )


def test_per_quote_path_sees_the_same_fixes():
    results = verify_citations(
        [
            "The hidden state $h_r$ is passed through softmax to obtain the attention weights",
            "This sentence appears nowhere in the source document at all, not once.",
        ],
        LATEX_SOURCE,
    )
    assert [r["ok"] for r in results] == [True, False]


# --- Why a quote failed -----------------------------------------------------
#
# The flag alone cannot tell an invented quote from one assembled out of real text
# in the wrong order, and those are different facts about a model. `reason` is what
# run-level statistics are read off, so the vocabulary is closed and every value has
# to be producible.

REASON_SOURCE = (
    "Participants were recruited from three universities over two semesters. "
    + "Filler sentence about the apparatus and the room here. " * 6
    + "All analyses were pre-registered before data collection began."
)

FAR_SOURCE = (
    "Participants were recruited from three universities over two semesters. "
    + "Filler sentence about the apparatus and the room here. " * 20
    + "All analyses were pre-registered before data collection began."
)


def reason(quote, source=REASON_SOURCE):
    return verify_citations([quote], source)[0]["reason"]


def test_a_contiguous_quote_is_ok():
    assert reason("Participants were recruited from three universities over two semesters.") == "ok"


def test_a_stitched_quote_says_it_was_stitched():
    assert reason(
        "Participants were recruited from three universities [...] All analyses were pre-registered"
    ) == "ok_ellipsis"


def test_an_invented_quote_is_absent():
    assert reason("Every participant was paid forty euros in cash on completion of the study.") == "absent"


def test_half_real_half_invented_is_partial():
    """The bucket a plain flag hides: real words with invented ones spliced in."""
    assert reason(
        "Participants were recruited from three universities and each was paid forty euros in cash"
    ) == "partial"


def test_real_text_in_the_wrong_order_is_reordered():
    assert reason(
        "All analyses were pre-registered before data collection ... "
        "Participants were recruited from three universities"
    ) == "reordered"


def test_real_text_too_far_apart_is_too_far():
    assert reason(
        "Participants were recruited from three universities ... "
        "All analyses were pre-registered before data collection",
        FAR_SOURCE,
    ) == "too_far"


def test_a_fragment_too_short_to_check_says_so():
    assert reason("three universities") == "skipped_short"


def test_every_reason_has_a_label_and_nothing_else_does():
    assert set(REASON_LABELS) == set(REASONS)


def test_a_verified_quote_carries_the_verifier_version():
    citations = build_citations(
        ["Participants were recruited from three universities over two semesters"],
        verify_citations(
            ["Participants were recruited from three universities over two semesters"], REASON_SOURCE
        ),
    )
    assert citations[0]["verifier"] == VERIFIER_VERSION
    assert citations[0]["reason"] in REASONS


def test_the_legacy_whole_blob_check_still_returns_one_flag():
    """Arbitration uses it, and one flag for a whole blob cannot carry a reason."""
    result = verify_citation(
        "Participants were recruited from three universities over two semesters.", REASON_SOURCE
    )
    assert result["ok"] is True
    assert "reason" not in result


# --- public matching primitives ---------------------------------------------
#
# SEER's citation locator answers "where is this quote" using these, and it only
# agrees with the verifier's "is this quote here" while they are the same code.
# Aliases, not copies -- so assert identity, which a reimplementation would fail.


def test_the_public_primitives_are_the_private_ones():
    assert verify_mod.normalize is verify_mod._normalize
    assert verify_mod.strip_edges is verify_mod._strip_edges
    assert verify_mod.fuzzy_find is verify_mod._fuzzy_find
    assert verify_mod.split_ellipsis == verify_mod._ELLIPSIS_RE.split


def test_the_public_normalize_takes_the_maths_pass():
    """The locator needs both spellings, same as the verifier's retry."""
    spaced = "the corpus contains $n = 2 6 2 1 4 4$ documents"
    assert verify_mod.normalize(spaced) != verify_mod.normalize(spaced, math=True)
