"""Tests for batching.resolve_groups."""

import pytest
from seer_annotator.config import Question, QuestionOption, RunConfig
from seer_annotator.batching import resolve_groups


def make_q(key: str, version_id: int) -> Question:
    return Question(
        question_id=version_id,
        key=key,
        version=1,
        version_id=version_id,
        label=key,
        help_text="",
        question_type="text",
    )


QS = [make_q("a", 1), make_q("b", 2), make_q("c", 3)]


def test_per_question():
    cfg = RunConfig(batching="per_question")
    groups = resolve_groups(cfg, QS)
    assert groups == [[QS[0]], [QS[1]], [QS[2]]]


def test_all():
    cfg = RunConfig(batching="all")
    groups = resolve_groups(cfg, QS)
    assert groups == [QS]


def test_size_1():
    cfg = RunConfig(batching={"size": 1})
    assert resolve_groups(cfg, QS) == [[QS[0]], [QS[1]], [QS[2]]]


def test_size_2():
    cfg = RunConfig(batching={"size": 2})
    groups = resolve_groups(cfg, QS)
    assert groups == [[QS[0], QS[1]], [QS[2]]]


def test_size_invalid():
    with pytest.raises(ValueError, match="size must be"):
        resolve_groups(RunConfig(batching={"size": 0}), QS)


def test_explicit_groups():
    cfg = RunConfig(batching=[["a", "b"], ["c"]])
    groups = resolve_groups(cfg, QS)
    assert groups == [[QS[0], QS[1]], [QS[2]]]


def test_explicit_partial_unlisted_become_solo():
    cfg = RunConfig(batching=[["a", "c"]])
    groups = resolve_groups(cfg, QS)
    assert groups[0] == [QS[0], QS[2]]
    assert groups[1] == [QS[1]]  # b not listed → solo


def test_explicit_unknown_key_errors():
    cfg = RunConfig(batching=[["a", "UNKNOWN"]])
    with pytest.raises(ValueError, match="unknown question key"):
        resolve_groups(cfg, QS)
