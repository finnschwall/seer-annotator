"""The OCR cache is keyed on the paper *and* the rendering, never the paper alone."""

import sqlite3

import pytest

from seer_annotator.seer_client import SeerClient
from seer_annotator.source_text import cached_full_text_for_run, full_text_for_run
from seer_annotator.store import Store


class RecordingClient(SeerClient):
    """A client that renders a paper differently for each run, like SEER's does."""

    def __init__(self, variants: dict[int, str], texts: dict[str, str]):
        super().__init__("https://seer.test/api", "tok")
        self._variants = variants
        self._texts = texts
        self.fetches: list[tuple[int, int]] = []

    def ocr_variant(self, run_id: int) -> str:
        return self._variants[run_id]

    async def fetch_ocr_markdown(self, paper_id: int, run_id: int):
        self.fetches.append((paper_id, run_id))
        return self._texts.get(self._variants[run_id])


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "state.db"))


def test_two_renderings_of_one_paper_do_not_collide(store):
    store.save_ocr(42, "body only", "no-refs")
    store.save_ocr(42, "body and references", "whole")

    assert store.get_ocr(42, "no-refs") == "body only"
    assert store.get_ocr(42, "whole") == "body and references"
    # A rendering nobody stored is a miss, not somebody else's text.
    assert store.get_ocr(42, "abstract-only") is None


def test_get_ocr_any_returns_some_rendering(store):
    assert store.get_ocr_any(42) is None
    store.save_ocr(42, "body only", "no-refs")
    assert store.get_ocr_any(42) == "body only"


@pytest.mark.asyncio
async def test_each_run_gets_its_own_rendering_and_is_fetched_once(store):
    client = RecordingClient(
        variants={1: "no-refs", 2: "whole", 3: "no-refs"},
        texts={"no-refs": "body only", "whole": "body and references"},
    )

    assert await full_text_for_run(client, store, 42, 1) == "body only"
    assert await full_text_for_run(client, store, 42, 2) == "body and references"
    # Run 3 renders like run 1, so it is served from run 1's entry.
    assert await full_text_for_run(client, store, 42, 3) == "body only"
    # Run 1 again: still cached.
    assert await full_text_for_run(client, store, 42, 1) == "body only"

    assert client.fetches == [(42, 1), (42, 2)]


@pytest.mark.asyncio
async def test_missing_text_is_cached_as_missing(store):
    client = RecordingClient(variants={1: "no-refs"}, texts={})

    assert await full_text_for_run(client, store, 42, 1) is None
    assert await full_text_for_run(client, store, 42, 1) is None
    # A paper with no text is asked for again — `None` is indistinguishable from
    # a miss, and that is deliberate: a paper whose text arrives later is picked
    # up on the next pass rather than staying blank for the life of the store.
    assert client.fetches == [(42, 1), (42, 1)]


def test_cached_reader_asks_for_the_runs_own_rendering(store):
    client = RecordingClient(
        variants={1: "no-refs", 2: "whole"},
        texts={"no-refs": "body only", "whole": "body and references"},
    )
    store.save_ocr(42, "body only", "no-refs")

    assert cached_full_text_for_run(client, store, 42, 1) == "body only"
    # Run 2's rendering was never stored, so there is nothing to verify against —
    # it does not silently borrow run 1's text.
    assert cached_full_text_for_run(client, store, 42, 2) is None
    assert client.fetches == []


def test_http_client_renders_one_way_for_every_run():
    client = SeerClient("https://seer.test/api", "tok")
    assert client.ocr_variant(1) == client.ocr_variant(2)


def test_store_written_before_renderings_is_brought_forward(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE ocr_cache (
            paper_id   INTEGER PRIMARY KEY,
            markdown   TEXT,
            fetched_at TEXT NOT NULL
        );
        INSERT INTO ocr_cache VALUES (42, 'text of unknown rendering', '2026-01-01T00:00:00Z');
        """
    )
    con.commit()
    con.close()

    store = Store(str(path))

    # The old schema recorded the text but not which rendering it was, so it is
    # served to nobody — and the row is not destroyed either.
    assert store.get_ocr(42, "") is None
    assert store.get_ocr(42, "no-refs") is None
    assert store.get_ocr_any(42) == "text of unknown rendering"

    # And the rebuilt table is the new one: two renderings now fit.
    store.save_ocr(42, "body only", "no-refs")
    store.save_ocr(42, "body and references", "whole")
    assert store.get_ocr(42, "no-refs") == "body only"
