# SPDX-License-Identifier: Apache-2.0
"""
Adaptive stream smoothing for speculative decoding.

Wraps an async iterator of SSE chunks and redistributes token timing
from bursty (speculative decode) to smooth character-by-character flow.

Algorithm:
  1. Warmup phase (first 1.5s): passthrough, just measure throughput
  2. After warmup: smooth to 90% of observed average chars/sec
  3. If buffer empties (model stalls): pause briefly, then resume
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator


class SmoothingIterator:
    """Async iterator wrapper that smooths SSE chunk emission timing.

    Tracks cumulative character throughput and adds delays between chunks
    to redistribute bursty speculative-decode output into a steady stream.
    """

    def __init__(
        self,
        source: AsyncIterator[str],
        warmup: float = 1.5,
        ratio: float = 0.90,
        empty_pause: float = 0.05,
    ):
        self._source = source
        self._warmup = warmup
        self._ratio = ratio
        self._empty_pause = empty_pause

    async def __aiter__(self):
        start_time = time.perf_counter()
        total_chars = 0
        next_emit_time: float | None = None

        source_iter = self._source.__aiter__()

        # Pull first chunk (role chunk — yield immediately, no smoothing)
        try:
            first_chunk = await source_iter.__anext__()
        except StopAsyncIteration:
            return

        yield first_chunk

        async for chunk in source_iter:
            now = time.perf_counter()
            elapsed = now - start_time
            chars_in_chunk = len(chunk)
            total_chars += chars_in_chunk

            # During warmup: passthrough, just accumulate stats
            if elapsed < self._warmup:
                yield chunk
                continue

            # Observed average throughput (chars/sec)
            avg_rate = total_chars / elapsed if elapsed > 0 else 1000
            # Target: emit at ratio of observed rate
            target_rate = avg_rate * self._ratio

            # Schedule this chunk's emission
            if next_emit_time is None:
                next_emit_time = now
            else:
                delay = chars_in_chunk / target_rate
                next_emit_time += delay

            # Sleep if we're ahead of schedule
            if next_emit_time > now:
                sleep_time = next_emit_time - now
                if sleep_time > 0.5:
                    # Model stalled — cap the sleep to avoid long freezes
                    await asyncio.sleep(self._empty_pause)
                else:
                    await asyncio.sleep(sleep_time)

            yield chunk
