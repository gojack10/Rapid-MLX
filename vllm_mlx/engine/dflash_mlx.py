# SPDX-License-Identifier: Apache-2.0
"""
DFlash-MLX speculative-decoding engine adapter for Rapid-MLX.

Wraps ``dflash_mlx`` (bstnxbt/dflash-mlx) as a drop-in engine that reuses
Rapid-MLX's chat-template application, tool-calling stack, and thinking
parsing while delegating token generation to the DFlash speculative decoder.

Enabled by passing ``--drafter <path>`` to ``rapid-mlx serve``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .batched import BatchedEngine, GenerationOutput

logger = logging.getLogger(__name__)


class DFlashMlxEngine(BatchedEngine):
    """BatchedEngine subclass that uses dflash-mlx for generation.

    Everything upstream of token generation (chat templates, tool processing,
    thinking-parser routing, message formatting) stays in Rapid-MLX.  Only
    the actual prompt-to-tokens step is replaced by dflash-mlx.
    """

    def __init__(
        self,
        model_name: str,
        *,
        drafter_path: str,
        scheduler_config=None,
        stream_interval: int = 1,
        force_mllm: bool = False,
        gpu_memory_utilization: float = 0.90,
    ):
        self._drafter_path = drafter_path
        self._dflash_ready = False
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_tokens_saved: int = 0
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_accepted: int = 0
        # Shut off BatchedEngine's scheduler/cache infra — DFlash manages
        # its own KV cache and prefix snapshots.
        if scheduler_config is None:
            from ..scheduler import SchedulerConfig

            scheduler_config = SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=False,
                use_memory_aware_cache=False,
            )
        else:
            scheduler_config.enable_prefix_cache = False
            scheduler_config.use_memory_aware_cache = False
        super().__init__(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # ------------------------------------------------------------------
    # Startup — skip BatchedEngine's scheduler/cache init
    # ------------------------------------------------------------------

    async def start(self):
        """Start the engine — load models, skip scheduler init."""
        if self._loaded:
            return
        await self._start_llm()
        self._loaded = True

    async def _start_llm(self) -> None:
        """Load target + drafter via dflash-mlx instead of plain mlx-lm."""
        import asyncio
        import concurrent.futures

        loop = asyncio.get_running_loop()
        self._model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dflash-load"
        )
        await loop.run_in_executor(
            self._model_load_executor, self._load_models_in_thread
        )

    def _load_models_in_thread(self) -> None:
        """Load target model, tokenizer, and DFlash drafter on worker thread."""
        import gc

        from dflash_mlx.generate import load_runtime_components

        logger.info(
            "[DFlash-MLX] Loading target=%s  drafter=%s",
            self._model_name,
            self._drafter_path,
        )

        # Build runtime context first so we get a consistent VerifyConfig
        from dflash_mlx.runtime_context import (
            build_runtime_context,
            runtime_config_from_profile,
            with_metal_limits,
        )

        runtime_config = runtime_config_from_profile("balanced")
        self._runtime_context = build_runtime_context(
            runtime_config, diagnostics_config=None
        )
        self._runtime_context = with_metal_limits(self._runtime_context, None)

        target_model, tokenizer, draft_model, _resolved = load_runtime_components(
            model_ref=self._model_name,
            draft_ref=self._drafter_path,
            draft_quant=None,
            verify_config=self._runtime_context.verify,
        )

        self._target_model = target_model
        self._draft_model = draft_model
        self._tokenizer = tokenizer
        self._model = target_model  # for route compatibility
        self._loaded = True
        self._dflash_ready = True

        # Log DFlash config banner
        rc = self._runtime_context.runtime
        logger.info(
            "[DFlash-MLX] Config: profile=%s verify=%s "
            "draft_sink=%s draft_window=%s "
            "prefix_cache=%s max_snapshot=%s "
            "prefill_step=%s",
            getattr(rc, "profile", "balanced"),
            getattr(rc, "verify_mode", "auto"),
            getattr(rc, "draft_sink_size", 64),
            getattr(rc, "draft_window_size", 1024),
            getattr(rc, "prefix_cache", False),
            getattr(rc, "max_snapshot_tokens", 24000),
            getattr(rc, "prefill_step_size", 4096),
        )

        gc.collect()
        logger.info("[DFlash-MLX] Models loaded, speculative decoding ready.")

    # ------------------------------------------------------------------
    # Properties expected by routes
    # ------------------------------------------------------------------

    @property
    def is_mllm(self) -> bool:
        return False

    @property
    def tokenizer(self):
        if not self._loaded:
            raise RuntimeError("Engine not loaded")
        return self._tokenizer

    @property
    def model(self):
        if not self._loaded:
            raise RuntimeError("Engine not loaded")
        return self._target_model

    @property
    def preserve_native_tool_format(self) -> bool:
        return getattr(self, "_preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self._preserve_native_tool_format = value

    # ------------------------------------------------------------------
    # Chat template — reused from BatchedEngine
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply the Rapid-MLX chat template (same as BatchedEngine)."""
        from ..api.tool_calling import convert_tools_for_template

        template_tools = convert_tools_for_template(tools) if tools else None
        return self._apply_chat_template(
            messages, template_tools, enable_thinking=enable_thinking
        )

    # ------------------------------------------------------------------
    # Generation — delegated to dflash-mlx
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Non-streaming generation via dflash-mlx."""
        output_text = ""
        completion_tokens = 0
        finish_reason = "stop"

        async for chunk in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            images=images,
            videos=videos,
            **kwargs,
        ):
            output_text = chunk.text
            completion_tokens = chunk.completion_tokens
            finish_reason = chunk.finish_reason or "stop"

        prompt_tokens = self._count_prompt_tokens(prompt)
        return GenerationOutput(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> Any:  # AsyncIterator[GenerationOutput]
        """Streaming generation via dflash-mlx with full telemetry."""
        import asyncio

        if not self._dflash_ready:
            raise RuntimeError("DFlash-MLX engine not loaded")

        tokenizer = self._tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        # Resolve stop token IDs
        stop_token_ids: set[int] = set()
        stop_strs = list(stop or [])
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)
        for s in stop_strs:
            ids = tokenizer.encode(s)
            stop_token_ids.update(ids)

        # Build prefix cache flow for this request
        prefix_flow = self._build_prefix_flow(tokenizer, prompt_ids)

        from dflash_mlx.runtime import stream_dflash_generate

        loop = asyncio.get_running_loop()

        # We use a queue so that dflash-mlx snapshot events are processed
        # on the MLX worker thread (where the stream is bound) rather than
        # on the asyncio event loop.
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        gen_done = loop.create_future()

        def _run_gen() -> None:
            """Run dflash-mlx generation on worker thread."""
            try:
                event_iter = stream_dflash_generate(
                    target_model=self._target_model,
                    tokenizer=self._tokenizer,
                    draft_model=self._draft_model,
                    prompt="",
                    max_new_tokens=max_tokens,
                    use_chat_template=False,
                    stop_token_ids=stop_token_ids,
                    prompt_tokens_override=prompt_ids,
                    prefix_snapshot=prefix_flow.get("snapshot"),
                    stable_prefix_len=prefix_flow.get("stable_prefix_len"),
                    prefix_cache=prefix_flow.get("cache"),
                    runtime_context=self._runtime_context,
                )
                # Process snapshot events on the worker thread while the
                # MLX stream is still active (inside the generator context).
                handler = prefix_flow.get("handler")
                for event in event_iter:
                    event_name = event.get("event", "")
                    if event_name == "prefill_snapshot_ready" and handler:
                        handler.handle_prefill_snapshot(event)
                    elif event_name == "generation_snapshot_ready" and handler:
                        handler.handle_generation_snapshot(event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
                loop.call_soon_threadsafe(gen_done.set_result, True)
            except Exception as exc:
                loop.call_soon_threadsafe(gen_done.set_exception, exc)

        self._model_load_executor.submit(_run_gen)

        # Consume events and yield GenerationOutput chunks from the asyncio side
        full_text = ""
        completion_tokens = 0
        finish_reason = "stop"
        detokenizer = (
            tokenizer.detokenizer
            if hasattr(tokenizer, "detokenizer")
            else tokenizer
        )
        if hasattr(detokenizer, "reset"):
            detokenizer.reset()

        dflash_accepted = 0
        dflash_cycles = 0
        dflash_elapsed_us = 0
        dflash_prefill_us = 0
        dflash_draft_us = 0
        dflash_verify_us = 0
        dflash_acceptance = 0.0

        while True:
            # Wait for next event or generation complete
            done, _ = await asyncio.wait(
                [loop.create_task(event_queue.get())],
                timeout=0.1,
            )
            if done:
                event = await done.pop()
            else:
                if gen_done.done():
                    # Generator finished, drain remaining events
                    await gen_done  # propagate any exception
                    while not event_queue.empty():
                        event = event_queue.get_nowait()
                        # process remaining events below
                        event_name = event.get("event", "")
                        if event_name == "token":
                            token_id = int(event.get("token_id", 0))
                            try:
                                token_text = detokenizer.decode([token_id])
                            except Exception:
                                token_text = ""
                            full_text += token_text
                            completion_tokens = int(event.get("generated_tokens", completion_tokens + 1))
                        elif event_name == "summary":
                            dflash_accepted = int(event.get("accepted_from_draft", 0))
                            dflash_cycles = int(event.get("cycles_completed", 0))
                            dflash_elapsed_us = float(event.get("elapsed_us", 0))
                            dflash_acceptance = float(event.get("acceptance_ratio", 0))
                            phase = event.get("phase_timings_us", {})
                            if isinstance(phase, dict):
                                dflash_prefill_us = float(phase.get("prefill", 0))
                                dflash_draft_us = float(phase.get("draft", 0))
                                dflash_verify_us = float(phase.get("verify", 0))
                    break
                continue

            event_name = event.get("event", "")
            if event_name == "token":
                token_id = int(event.get("token_id", 0))
                try:
                    token_text = detokenizer.decode([token_id])
                except Exception:
                    token_text = (
                        tokenizer.decode([token_id])
                        if hasattr(tokenizer, "decode")
                        else ""
                    )
                full_text += token_text
                completion_tokens = int(
                    event.get("generated_tokens", completion_tokens + 1)
                )

                if token_id in stop_token_ids:
                    finish_reason = "stop"

                yield GenerationOutput(
                    text=full_text,
                    new_text=token_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=False,
                    finish_reason=None,
                )

            elif event_name == "summary":
                completion_tokens = int(
                    event.get("generation_tokens", completion_tokens)
                )
                finish_reason = event.get("finish_reason", "stop")
                dflash_accepted = int(event.get("accepted_from_draft", 0))
                dflash_cycles = int(event.get("cycles_completed", 0))
                dflash_elapsed_us = float(event.get("elapsed_us", 0))
                dflash_acceptance = float(event.get("acceptance_ratio", 0))
                phase = event.get("phase_timings_us", {})
                if isinstance(phase, dict):
                    dflash_prefill_us = float(phase.get("prefill", 0))
                    dflash_draft_us = float(phase.get("draft", 0))
                    dflash_verify_us = float(phase.get("verify", 0))

            # Snapshot events were handled on the worker thread already.

        # Update engine-level stats
        self._total_requests += 1
        self._total_tokens += completion_tokens
        self._total_accepted += dflash_accepted
        if prefix_flow.get("hit_tokens", 0) > 0:
            self._cache_hits += 1
            self._cache_tokens_saved += prefix_flow.get("hit_tokens", 0)
        else:
            self._cache_misses += 1

        # Emit detailed DFlash telemetry log
        cache_tag = (
            f"HIT({prefix_flow.get('hit_tokens', 0)}/{prompt_tokens})"
            if prefix_flow.get("hit_tokens", 0) > 0
            else "MISS"
        )
        tps = completion_tokens / (dflash_elapsed_us / 1_000_000) if dflash_elapsed_us > 0 else 0
        logger.info(
            "[DFlash-MLX] request: %d tokens  %.1f tok/s  "
            "accept=%.1f%%  accepted=%d/%d  cycles=%d  "
            "prefill=%.0fms  draft=%.0fms  verify=%.0fms  "
            "cache=%s  lookup=%.2fms",
            completion_tokens,
            tps,
            dflash_acceptance * 100,
            dflash_accepted,
            completion_tokens,
            dflash_cycles,
            dflash_prefill_us / 1000,
            dflash_draft_us / 1000,
            dflash_verify_us / 1000,
            cache_tag,
            prefix_flow.get("lookup_ms", 0),
        )

        yield GenerationOutput(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finished=True,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Prefix cache integration
    # ------------------------------------------------------------------

    def _build_prefix_flow(
        self, tokenizer: Any, prompt_ids: list[int]
    ) -> dict[str, Any]:
        """Create a dflash-mlx PrefixCacheFlow and return its key fields.

        Returns a dict with: cache, snapshot, stable_prefix_len, hit_tokens,
        lookup_ms, handler (the PrefixCacheFlow instance for post-generation
        snapshot insertion).
        """
        try:
            from dflash_mlx.server.prefix_cache_flow import (
                PrefixCacheFlow,
                compute_stable_prefix_len,
                get_dflash_prefix_cache,
            )
            from dflash_mlx.server.prefix_cache_manager import (
                build_prefix_key,
                chat_template_marker_ids,
            )

            cache = get_dflash_prefix_cache(self._runtime_context)
            if cache is None:
                logger.debug("[DFlash-MLX] prefix cache disabled in runtime config")
                return {
                    "cache": None,
                    "snapshot": None,
                    "stable_prefix_len": None,
                    "hit_tokens": 0,
                    "lookup_ms": 0,
                    "handler": None,
                }

            # Build a minimal model_provider stub for build_prefix_key
            class _StubProvider:
                model_key = (self._model_name, None, self._drafter_path)

            key = build_prefix_key(
                _StubProvider(), self._draft_model, self._runtime_context
            )

            im_start_id, assistant_id = chat_template_marker_ids(tokenizer)
            stable_prefix_len = compute_stable_prefix_len(
                prompt_ids,
                im_start_id=im_start_id,
                assistant_id=assistant_id,
            )

            lookup_tokens = prompt_ids[:stable_prefix_len]
            logger.debug(
                "[DFlash-MLX] cache lookup: prompt=%d stable=%d "
                "lookup_len=%d first_tokens=%s",
                len(prompt_ids),
                stable_prefix_len,
                len(lookup_tokens),
                lookup_tokens[:8],
            )
            lookup_t0 = time.perf_counter_ns()
            matched_len, snapshot = cache.lookup(lookup_tokens, key)
            lookup_ms = (time.perf_counter_ns() - lookup_t0) / 1e6
            hit_tokens = int(matched_len)

            if hit_tokens > 0:
                c_entries = len(getattr(cache, "_entries", {}))
                c_stats = getattr(cache, "_stats", {})
                logger.info(
                    "[DFlash-MLX] prefix cache HIT %d/%d tokens "
                    "(stable_prefix=%d, entries=%d, hits=%d, misses=%d, lookup=%.2fms)",
                    hit_tokens, len(prompt_ids), stable_prefix_len, c_entries,
                    c_stats.get("exact_hits",0)+c_stats.get("prefix_hits",0),
                    c_stats.get("misses",0), lookup_ms)
            else:
                c_entries = len(getattr(cache, "_entries", {}))
                c_stats = getattr(cache, "_stats", {})
                logger.info(
                    "[DFlash-MLX] prefix cache MISS "
                    "(prompt=%d, stable_prefix=%d, entries=%d, "
                    "hits=%d, misses=%d, fingerprint_rejects=%d, lookup=%.2fms)",
                    len(prompt_ids), stable_prefix_len, c_entries,
                    c_stats.get("exact_hits",0)+c_stats.get("prefix_hits",0),
                    c_stats.get("misses",0), c_stats.get("fingerprint_rejects",0),
                    lookup_ms)

            flow = PrefixCacheFlow(
                cache=cache,
                key=key,
                stable_prefix_len=stable_prefix_len,
                snapshot=snapshot,
                lookup_ms=lookup_ms,
                hit_tokens=hit_tokens,
                draft_model=self._draft_model,
                runtime_context=self._runtime_context,
            )

            return {
                "cache": cache,
                "snapshot": snapshot,
                "stable_prefix_len": stable_prefix_len,
                "hit_tokens": hit_tokens,
                "lookup_ms": lookup_ms,
                "handler": flow,
            }
        except Exception as exc:
            logger.debug("[DFlash-MLX] prefix cache init failed (non-fatal): %s", exc)
            return {
                "cache": None,
                "snapshot": None,
                "stable_prefix_len": None,
                "hit_tokens": 0,
                "lookup_ms": 0,
                "handler": None,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Count tokens in a prompt string."""
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return len(tokenizer.encode(prompt))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        base = super().get_stats()
        base["engine_type"] = "dflash-mlx"
        base["dflash_ready"] = self._dflash_ready
        base["drafter_path"] = self._drafter_path
        base["dflash_requests"] = self._total_requests
        base["dflash_total_tokens"] = self._total_tokens
        base["dflash_accepted_drafts"] = self._total_accepted
        base["dflash_cache_hits"] = self._cache_hits
        base["dflash_cache_misses"] = self._cache_misses
        base["dflash_cache_tokens_saved"] = self._cache_tokens_saved
        if self._total_requests > 0:
            base["dflash_avg_acceptance"] = (
                self._total_accepted / max(self._total_tokens, 1)
            )
            base["dflash_cache_hit_rate"] = self._cache_hits / self._total_requests
        return base
