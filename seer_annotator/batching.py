"""Resolve run batching config -> list[list[Question]]."""

from __future__ import annotations

from .config import BatchingConfig, Question, RunConfig


def resolve_groups(run_config: RunConfig, questions: list[Question]) -> list[list[Question]]:
    """Return ordered groups of questions based on the run's batching config."""
    batching = run_config.batching
    key_to_q = {q.key: q for q in questions}

    if batching == "per_question":
        return [[q] for q in questions]

    if batching == "all":
        return [list(questions)]

    if isinstance(batching, dict):
        size = batching["size"]
        if size < 1:
            raise ValueError(f"batching size must be >= 1, got {size}")
        return [questions[i : i + size] for i in range(0, len(questions), size)]

    if isinstance(batching, list):
        # Explicit groups by key; any key not listed becomes its own solo group.
        used: set[str] = set()
        groups: list[list[Question]] = []

        for key_group in batching:
            for k in key_group:
                if k not in key_to_q:
                    raise ValueError(
                        f"batching references unknown question key {k!r}. "
                        f"Known keys: {list(key_to_q)}"
                    )
            groups.append([key_to_q[k] for k in key_group])
            used.update(key_group)

        # Remaining questions each get their own group, preserving JSON order.
        for q in questions:
            if q.key not in used:
                groups.append([q])

        return groups

    raise ValueError(f"Unrecognised batching config: {batching!r}")
