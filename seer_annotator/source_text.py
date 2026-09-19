"""One definition of "what full text does this run see for this paper".

Both orchestrators, and the prompt previews in `cli.py`, resolve source text the
same way: ask the local store, and on a miss ask the client and cache what comes
back. The part that is easy to get wrong is the key. Source text is rendered
*per run* — SEER withholds the sections a run's `exclude_sections` names — so a
cache keyed on the paper alone hands the second run of a job whatever the first
run asked for, silently. The key is the paper plus the client's own name for the
rendering (`SeerClient.ocr_variant`), which is what these helpers enforce by
never taking one without the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .seer_client import SeerClient
    from .store import Store


async def full_text_for_run(
    client: "SeerClient", store: "Store", paper_id: int, run_id: int
) -> str | None:
    """Return the full text `run_id` should be sent for `paper_id`, or None.

    Served from the job store when that run's rendering of the paper is already
    there, otherwise fetched from the client and cached under that rendering. A
    miss is cached too: `None` records "asked, nothing to have".
    """
    variant = client.ocr_variant(run_id)
    text = store.get_ocr(paper_id, variant)
    if text is not None:
        return text
    text = await client.fetch_ocr_markdown(paper_id, run_id)
    store.save_ocr(paper_id, text, variant)
    return text


def cached_full_text_for_run(
    client: "SeerClient", store: "Store", paper_id: int, run_id: int
) -> str | None:
    """The store-only half of `full_text_for_run`, for passes that never fetch.

    Pass 2 and reformatting re-read text to verify quotes against it; they run on
    work Pass 1 already did, so the text they want is the rendering that run was
    sent, and if it is not in the store there is nothing to verify against.
    """
    return store.get_ocr(paper_id, client.ocr_variant(run_id))
