# SPDX-License-Identifier: Apache-2.0
"""
Adaptive stream smoothing for speculative decoding.

Parses incoming SSE chunks, extracts text content, and re-emits
character-by-character at a smooth adaptive rate.

Algorithm:
  1. Warmup phase (first 1.5s): passthrough, just measure throughput
  2. After warmup: smooth to 90% of observed average chars/sec,
     emitting one character per SSE chunk
  3. If buffer empties (model stalls): pause briefly, then resume
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator


class SmoothingIterator:
    """Async iterator that parses SSE chunks and re-emits char-by-char.

    Takes bursty speculative-decode output and redistributes it into
    a steady character-by-character stream.
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

    # ------------------------------------------------------------------
    # SSE chunk parsing / re-emission
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sse(raw: str) -> dict | None:
        """Parse a ``data: {...}`` SSE line into a dict, or return None."""
        line = raw.strip()
        if not line.startswith("data: "):
            return None
        try:
            return json.loads(line[6:])
        except (json.JSONDecodeError, KeyError):
            return None

    def _char_chunk(self, template: dict, field: str, ch: str) -> str:
        """Build a full SSE chunk for a single character, preserving
        id/object/created/model from the original template."""
        # Deep-copy the template structure
        out = {
            "id": template.get("id", ""),
            "object": template.get("object", "chat.completion.chunk"),
            "created": template.get("created", 0),
            "model": template.get("model", ""),
            "choices": [{"index": 0, "delta": {field: ch}}],
        }
        return f"data: {json.dumps(out, ensure_ascii=False, separators=(',', ':'))}\n\n"

    def _extract_text(self, chunk: dict) -> tuple[str, str]:
        """Extract (content, reasoning_content) from choices[0].delta."""
        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return "", ""
        delta = choices[0].get("delta", {})
        return delta.get("content") or "", delta.get("reasoning_content") or ""

    def _has_finish(self, chunk: dict) -> bool:
        """Check if this chunk carries a finish_reason (terminal chunk)."""
        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list):
            return "finish_reason" in choices[0]
        return False

    # ------------------------------------------------------------------
    # Main iterator
    # ------------------------------------------------------------------

    async def __aiter__(self):
        start_time = time.perf_counter()
        total_chars = 0
        next_emit_time: float | None = None
        smoothing_active = False

        # Template captured from first content chunk (id, model, created, etc.)
        template: dict | None = None

        source_iter = self._source.__aiter__()

        # --- Pull first chunk (role chunk — yield verbatim) ---
        try:
            first_raw = await source_iter.__anext__()
        except StopAsyncIteration:
            return
        yield first_raw

        # --- Process remaining chunks ---
        async for raw in source_iter:
            now = time.perf_counter()
            elapsed = now - start_time

            # Pass through terminal markers
            if "[DONE]" in raw:
                yield raw
                continue

            chunk = self._parse_sse(raw)
            if chunk is None:
                yield raw
                continue

            # Terminal chunk (finish_reason, usage) — pass through
            if self._has_finish(chunk):
                yield raw
                continue

            content_text, reasoning_text = self._extract_text(chunk)
            if not content_text and not reasoning_text:
                # Non-text chunk (e.g. tool_call metadata) — pass through
                yield raw
                continue

            # Capture template from first content chunk
            if template is None:
                template = dict(chunk)

            # Build text to emit (reasoning first, then content)
            text_to_emit = reasoning_text + content_text
            field = "reasoning_content" if reasoning_text else "content"

            # --- Emit character by character ---
            for ch in text_to_emit:
                total_chars += 1
                now = time.perf_counter()
                elapsed = now - start_time

                # Warmup: passthrough
                if elapsed < self._warmup and not smoothing_active:
                    yield self._char_chunk(template, field, ch)
                    continue

                # Activate smoothing
                smoothing_active = True

                # Target rate: ratio of observed average
                avg_rate = total_chars / elapsed if elapsed > 0 else 1000
                target_rate = avg_rate * self._ratio

                # Schedule emission
                if next_emit_time is None:
                    next_emit_time = now
                else:
                    next_emit_time += 1.0 / target_rate

                # Sleep if ahead of schedule
                if next_emit_time > now:
                    sleep_time = next_emit_time - now
                    if sleep_time > 0.5:
                        await asyncio.sleep(self._empty_pause)
                    else:
                        await asyncio.sleep(sleep_time)

                yield self._char_chunk(template, field, ch)
