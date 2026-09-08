"""Whether a model's reply is a *finished* reply.

A cut-off reply is still text, and a cut-off JSON reply still parses once
``json_repair`` closes the braces it never got to write. Nothing downstream can
tell that apart from a model that simply had less to say, so the judgment has to
be made here, from the reply itself, before anything reads values out of it.

Pass 1 has been checked this way since ``_demote_truncated_p1``; Pass 2 was not,
and that is what this module adds. The two failure modes it names want opposite
things from the reader, which is why they are separate codes:

``pass2_truncated``
    The model had more to say and ran out of room. Raising the ceiling helps.

``pass2_degenerate_output``
    The model stopped producing language and looped on a word or phrase until
    the ceiling stopped it. Raising the ceiling only lets the loop run longer;
    retrying, or sending Pass 2 to a different model, is what recovers the cell.

Kept in its own module with no imports beyond ``re`` so that a display path can
import it without pulling in the provider clients that ``batch_runner`` needs.
"""

import re

# Finish reasons that mean "the model had more to say and ran out of room", so
# the text is cut off mid-sentence and is not a complete answer. "length" is
# OpenAI's and LiteLLM's normalized value; "max_tokens" and
# "model_context_window_exceeded" are Anthropic `stop_reason`s. Every other
# reason ("end_turn", "stop", "tool_use", "stop_sequence", ...) means the model
# finished on its own terms.
TRUNCATED_FINISH_REASONS = frozenset({
    "length", "max_tokens", "model_context_window_exceeded",
})

# --- Degenerate output ("the model got stuck repeating itself") -------------
#
# Two signals, either one enough:
#
#   _MIN_REPEAT_RUN     — longest run of one identical token repeated back to
#                         back. Catches a single-word loop at any length.
#   _MAX_TAIL_UNIQUENESS — distinct fraction of the last _TAIL_TOKENS tokens.
#                         Catches a loop over a repeating *phrase*, where no
#                         single token repeats consecutively at all.
#
# The thresholds were measured against one deployment's stored replies rather
# than picked, and no other deployment's replies have been checked. On text that
# was not degenerate the longest repeat run was 4 and the lowest tail uniqueness
# 0.365 (20,000 Pass-1 replies, and every stored Pass-2 reply that was not
# itself a loop); on the loops the run was 7,500+ and uniqueness 0.005. Nothing
# sat in between. The margins are wide in both directions, which is the reason
# to expect them to travel — not evidence that they have.
_MIN_REPEAT_RUN = 20
_TAIL_TOKENS = 200
_MIN_TAIL_TOKENS = 60
_MAX_TAIL_UNIQUENESS = 0.15

# How much of the end of a reply is enough to judge it. A loop runs to the end
# of the text by definition, so the tail is the only part that has to be read —
# which is what lets a caller slice this much off in SQL instead of loading
# replies that are tens of kilobytes each.
DEGENERATE_TAIL_CHARS = 4000


def looks_degenerate(text: "str | None") -> bool:
    """True when `text` ends in a repetition loop rather than in language.

    Safe to call on a tail slice: only the end of a reply is examined, and a
    slice too short to judge returns False. Fails closed on purpose — calling a
    real answer a loop is worse than missing one, since the loop still surfaces
    as some other failure.
    """
    words = re.findall(r'\S+', (text or '')[-DEGENERATE_TAIL_CHARS:].lower())
    if not words:
        return False

    longest = run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        longest = max(longest, run)
    if longest >= _MIN_REPEAT_RUN:
        return True

    tail_size = min(_TAIL_TOKENS, len(words))
    if tail_size < _MIN_TAIL_TOKENS:
        return False
    tail = words[-tail_size:]
    return len(set(tail)) / tail_size < _MAX_TAIL_UNIQUENESS


def pass2_incomplete_reason(
    text: "str | None", usage: "dict | None",
) -> "tuple[str, str] | None":
    """``(code, detail)`` when this Pass-2 reply did not finish, else None.

    Call it on the raw reply before parsing. A reply that fails here must fail
    the whole question group: Pass 2 answers a group in one JSON document, and
    the items that got out before it stopped are only the ones that happened to
    come first in the schema's field order, not the ones that were right.

    The order matters. A loop that ran to the ceiling carries exactly the
    evidence of a cut-off answer, so it is checked first or it would be reported
    as a budget problem and get the opposite advice.

    The last rule is the fallback for a provider that reports no finish reason
    at all, where spending the entire ceiling is the only evidence left. It
    needs ``max_tokens`` in ``usage``: the online path records the ceiling it
    actually sent, the batch path does not, so on batch a silent provider is not
    caught here — the parser's own completeness check is the backstop.
    """
    usage = usage or {}

    if looks_degenerate(text):
        return (
            "pass2_degenerate_output",
            "pass2 degenerate output — the format model repeated itself instead of "
            "writing JSON, so its reply is not a complete result document. Retry the "
            "cell, or send Pass 2 to a different model. Raising the Pass 2 output "
            "ceiling does not help: the ceiling is what stopped the loop.",
        )

    reason = usage.get("finish_reason")
    if reason in TRUNCATED_FINISH_REASONS:
        return (
            "pass2_truncated",
            f"pass2 truncated — the format model ran out of output budget "
            f"(finish_reason={reason!r}), so its JSON stops mid-object and the items "
            "it did write may be missing their later fields. Raise the run's Pass 2 "
            "max output tokens and re-run.",
        )

    max_tokens = usage.get("max_tokens")
    output_tokens = usage.get("output_tokens")
    if not reason and max_tokens and output_tokens and output_tokens >= max_tokens:
        return (
            "pass2_truncated",
            f"pass2 truncated — the format model spent its entire output budget "
            f"({output_tokens} of {max_tokens} tokens) and the provider reported no "
            "finish reason, so its JSON is cut off. Raise the run's Pass 2 max output "
            "tokens and re-run.",
        )

    return None
