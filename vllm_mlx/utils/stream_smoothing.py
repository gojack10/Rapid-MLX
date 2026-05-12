# SPDX-License-Identifier: Apache-2.0
"""
Adaptive stream smoothing for speculative decoding.

Buffers bursty speculative-decode output and re-emits character-by-character
at the model's *measured* sustained speed.  A brief warmup fills a bucket;
after warmup, characters flow at exactly the observed rate so the visual
speed is always native to the loaded model.

Algorithm:
  1. Warmup: buffer characters, measure arrival rate
  2. After warmup: emit at the measured rate (100 %)
  3. If buffer runs dry (model fell behind): pause briefly
  4. If buffer overfills (model sped up / warmup underestimated):
     recalc rate from a rolling window
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


class SmoothingIterator:
    """Async iterator that buffers SSE chunks and re-emits char-by-char
    at the dynamically measured model speed."""

    def __init__(
        self,
        source: AsyncIterator[str],
        warmup_secs: float = 0.6,
        empty_pause: float = 0.04,
        rate_window_secs: float = 2.0,
        max_bucket_chars: int = 120,
    ):
        self._source = source
        self._warmup_secs = float(warmup_secs)
        self._empty_pause = float(empty_pause)
        self._rate_window = float(rate_window_secs)
        self._max_bucket = int(max_bucket_chars)

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sse(raw: str) -> dict | None:
        line = raw.strip()
        if not line.startswith("data: "):
            return None
        try:
            return json.loads(line[6:])
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _extract_text(chunk: dict) -> tuple[str, str, str | None]:
        """Return (content, reasoning, tool_call_name)."""
        choices = chunk.get("choices", [])
        if not choices or not isinstance(choices, list):
            return "", "", None
        delta = choices[0].get("delta", {})
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        tc = delta.get("tool_calls")
        if tc and isinstance(tc, list) and len(tc) > 0:
            fn = tc[0].get("function", {}) if isinstance(tc[0], dict) else None
            if fn and isinstance(fn, dict) and fn.get("name"):
                return "", "", fn["name"]
        return content, reasoning, None

    @staticmethod
    def _has_finish(chunk: dict) -> bool:
        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list):
            return "finish_reason" in choices[0]
        return False

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    @staticmethod
    def _build_char_sse(template: dict, field: str, ch: str) -> str:
        out = {
            "id": template.get("id", ""),
            "object": template.get("object", "chat.completion.chunk"),
            "created": template.get("created", 0),
            "model": template.get("model", ""),
            "choices": [{"index": 0, "delta": {field: ch}}],
        }
        return f"data: {json.dumps(out, ensure_ascii=False, separators=(',', ':'))}\n\n"

    # ------------------------------------------------------------------
    # Cleanup (propagate aclose/GeneratorExit to source)
    # ------------------------------------------------------------------

    async def aclose(self):
        """Close the source iterator. Called on client disconnect."""
        try:
            await self._source.aclose()
        except (GeneratorExit, StopAsyncIteration):
            pass

    # ------------------------------------------------------------------
    # Main iterator
    # ------------------------------------------------------------------

    async def __aiter__(self):
        # Each buffer item: (field, char, pre_rendered_sse)
        buffer: collections.deque[tuple[str, str, str]] = collections.deque()

        # Arrival-time ring buffer for rolling rate measurement.
        # Stores (timestamp, char_count) for recent arrivals.
        _arrivals: collections.deque[tuple[float, int]] = collections.deque()

        template: dict | None = None
        start_time = time.perf_counter()
        warmed = False
        done = False
        total_buffered = 0  # chars seen since start
        source_chunks = 0
        passthrough_chunks = 0
        emitted_chars = 0
        peak_buffer = 0
        sleep_ns = 0
        warmup_buffer_chars = 0
        first_emit_time: float | None = None
        last_emit_time: float | None = None

        # Current emission interval (seconds per char).  Initialised to a
        # conservative value until we have enough data.
        _emit_interval = 0.02  # 50 chars/s — will be overwritten after warmup

        def _record_arrival(n: int):
            nonlocal total_buffered
            now = time.perf_counter()
            total_buffered += n
            _arrivals.append((now, n))
            # Prune entries older than rate window
            cutoff = now - self._rate_window
            while _arrivals and _arrivals[0][0] < cutoff:
                _arrivals.popleft()

        def _current_rate() -> float:
            """Chars/sec over the rolling rate window."""
            if not _arrivals:
                return 15.0  # floor; avoid division-by-zero
            window_start = _arrivals[0][0]
            total_in_window = sum(n for _, n in _arrivals)
            elapsed = time.perf_counter() - window_start
            if elapsed <= 0.001:
                return 80.0
            return total_in_window / elapsed

        def _maybe_recalc_interval():
            nonlocal _emit_interval
            rate = _current_rate()
            if rate > 0:
                _emit_interval = 1.0 / rate

        def _buffer_text(text: str, field: str):
            nonlocal peak_buffer
            n = len(text)
            _record_arrival(n)
            for ch in text:
                sse = self._build_char_sse(template, field, ch) if template else ""
                buffer.append((field, ch, sse))
            peak_buffer = max(peak_buffer, len(buffer))

        def _mark_emit() -> None:
            nonlocal emitted_chars, first_emit_time, last_emit_time
            emitted_chars += 1
            now = time.perf_counter()
            if first_emit_time is None:
                first_emit_time = now
            last_emit_time = now

        source_iter = self._source.__aiter__()

        try:
            # --- First chunk (role) — pass through verbatim ---
            try:
                first_raw = await source_iter.__anext__()
            except StopAsyncIteration:
                return
            source_chunks += 1
            passthrough_chunks += 1
            yield first_raw

            # --- Process remaining chunks ---
            async for raw in source_iter:
                source_chunks += 1
                if done:
                    passthrough_chunks += 1
                    yield raw
                    continue

                if "[DONE]" in raw:
                    done = True
                    while buffer:
                        _, _, sse = buffer.popleft()
                        if sse:
                            _mark_emit()
                            yield sse
                    passthrough_chunks += 1
                    yield raw
                    continue

                chunk = self._parse_sse(raw)
                if chunk is None:
                    passthrough_chunks += 1
                    yield raw
                    continue

                if template is None:
                    template = dict(chunk)

                if self._has_finish(chunk):
                    while buffer:
                        _, _, sse = buffer.popleft()
                        if sse:
                            _mark_emit()
                            yield sse
                    passthrough_chunks += 1
                    yield raw
                    continue

                content, reasoning, tool_name = self._extract_text(chunk)
                if tool_name:
                    passthrough_chunks += 1
                    yield raw
                    continue

                text = reasoning or content
                field = "reasoning_content" if reasoning else "content"
                if not text:
                    passthrough_chunks += 1
                    yield raw
                    continue

                # --- Buffer incoming text ---
                _buffer_text(text, field)

                elapsed = time.perf_counter() - start_time

                # --- Warmup: just buffer, don't emit yet ---
                if not warmed and elapsed < self._warmup_secs:
                    continue

                # --- Activate ---
                if not warmed:
                    warmed = True
                    warmup_buffer_chars = len(buffer)
                    _maybe_recalc_interval()
                    logger.info(
                        "[STREAM-SMOOTH] activate warmup=%.2fs buffered=%d observed_cps=%.1f interval=%.2fms",
                        elapsed,
                        warmup_buffer_chars,
                        _current_rate(),
                        _emit_interval * 1000.0,
                    )

                # --- Drain buffer at measured rate ---
                while buffer:
                    # Recalc rate periodically (every ~20 chars or when bucket
                    # overfills / underfills)
                    if len(buffer) % 20 == 0 or len(buffer) > self._max_bucket or len(buffer) < 4:
                        _maybe_recalc_interval()

                    field, ch, sse = buffer.popleft()
                    if sse:
                        _mark_emit()
                        yield sse

                    # If buffer ran dry, wait for more input (skip sleep)
                    if not buffer:
                        break

                    # Pace emission at measured rate
                    remaining = _emit_interval
                    if remaining > 0 and remaining < 0.5:
                        _sleep_start = time.perf_counter_ns()
                        await asyncio.sleep(remaining)
                        sleep_ns += time.perf_counter_ns() - _sleep_start

            # --- Source exhausted; drain remaining buffer ---
            while buffer:
                _, _, sse = buffer.popleft()
                if sse:
                    _mark_emit()
                    yield sse

            total_elapsed = time.perf_counter() - start_time
            emit_elapsed = (
                (last_emit_time - first_emit_time)
                if first_emit_time is not None and last_emit_time is not None and last_emit_time > first_emit_time
                else 0.0
            )
            logger.info(
                "[STREAM-SMOOTH] source_chunks=%d passthrough=%d input_chars=%d emitted_chars=%d "
                "peak_buffer=%d warmup_buffer=%d emit_cps=%.1f total_cps=%.1f sleep=%.1fms warmup=%.2fs "
                "rate_window=%.2fs max_bucket=%d",
                source_chunks,
                passthrough_chunks,
                total_buffered,
                emitted_chars,
                peak_buffer,
                warmup_buffer_chars,
                emitted_chars / emit_elapsed if emit_elapsed > 0 else 0.0,
                emitted_chars / total_elapsed if total_elapsed > 0 else 0.0,
                sleep_ns / 1e6,
                self._warmup_secs,
                self._rate_window,
                self._max_bucket,
            )

        except GeneratorExit:
            logger.warning(
                f"[STREAM-SMOOTH] ** GeneratorExit after "
                f"source_chunks={source_chunks} emitted_chars={emitted_chars}, buffer_remaining={len(buffer)}"
            )
            raise
        except asyncio.CancelledError:
            logger.warning(
                f"[STREAM-SMOOTH] ** CancelledError after "
                f"source_chunks={source_chunks} emitted_chars={emitted_chars}, buffer_remaining={len(buffer)}"
            )
            # Close source so GeneratorExit propagates to engine
            await self._source.aclose()
            raise
