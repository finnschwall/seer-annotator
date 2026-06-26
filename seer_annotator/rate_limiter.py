"""Per-provider sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import collections
import time


class PerProviderRateLimiter:
    """Sliding-window rate limiter keyed by provider string.

    rpm=None disables limiting entirely (all acquire() calls return immediately).
    Each provider gets its own lock so providers don't block each other.
    """

    def __init__(self, rpm: float | None) -> None:
        self._rpm = rpm
        self._window = 60.0
        self._timestamps: dict[str, collections.deque[float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, provider: str) -> None:
        if self._rpm is None:
            return
        if provider not in self._locks:
            self._locks[provider] = asyncio.Lock()
        async with self._locks[provider]:
            if provider not in self._timestamps:
                self._timestamps[provider] = collections.deque()
            ts = self._timestamps[provider]
            now = time.monotonic()
            while ts and now - ts[0] >= self._window:
                ts.popleft()
            if len(ts) >= self._rpm:
                sleep_for = self._window - (now - ts[0])
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while ts and now - ts[0] >= self._window:
                    ts.popleft()
            ts.append(now)
