# SPDX-License-Identifier: Apache-2.0
"""
DFlash-MLX speculative-decoding engine adapter for Rapid-MLX.
"""

from __future__ import annotations

import logging
import time
import asyncio
from pathlib import Path
from typing import Any

from .batched import BatchedEngine, GenerationOutput

logger = logging.getLogger(__name__)


class DFlashMlxEngine(BatchedEngine):
    """BatchedEngine subclass that uses dflash-mlx for generation."""

    def __init__(
        self,
        model_name: str,
        *,
        drafter_path: str,
        scheduler_config=None,
        stream_interval: int = 1,
        force_mllm: bool = False,
        gpu_memory_utilization: float = 0.90,
        dflash_config: Any | None = None,
    ):
        self._drafter_path = drafter_path
        self._dflash_config = dflash_config
        self._dflash_ready = False
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_tokens_saved: int = 0
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_accepted: int = 0
        if scheduler_config is None:
            from ..scheduler import SchedulerConfig
            scheduler_config = SchedulerConfig(
                max_num_seqs=1,
                enable_prefix_cache=True,
                use_memory_aware_cache=True,
            )
        # Keep Rapid-MLX's own prefix cache active — it handles the
        # conversation-level cache that DFlash's internal cache struggles with.
        super().__init__(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # ------------------------------------------------------------------ startup
    async def start(self):
        if self._loaded:
            return
        await self._start_llm()
        self._loaded = True

    async def _start_llm(self) -> None:
        import concurrent.futures
        loop = asyncio.get_running_loop()
        self._model_load_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dflash-load"
        )
        await loop.run_in_executor(self._model_load_executor, self._load_models_in_thread)

    def _load_models_in_thread(self) -> None:
        import gc
        from dflash_mlx.generate import load_runtime_components
        from dflash_mlx.runtime_context import (
            build_runtime_context, runtime_config_from_profile, with_metal_limits,
        )
        from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
        from dflash_mlx.metal_limits import apply_metal_limits, parse_memory_limit

        logger.info("[DFlash-MLX] Loading target=%s  drafter=%s", self._model_name, self._drafter_path)

        cfg = self._dflash_config
        profile = getattr(cfg, "profile", None) or "balanced"
        prefill_step_size = (
            getattr(cfg, "prefill_step_size", None)
            if getattr(cfg, "_dflash_prefill_step_size_explicit", False)
            else None
        )
        runtime_config = runtime_config_from_profile(
            profile=profile,
            prefill_step_size=prefill_step_size,
            draft_sink_size=getattr(cfg, "draft_sink_size", None),
            draft_window_size=getattr(cfg, "draft_window_size", None),
            verify_len_cap=getattr(cfg, "verify_len_cap", None),
            prefix_cache=getattr(cfg, "prefix_cache", None),
            prefix_cache_max_entries=getattr(cfg, "prefix_cache_max_entries", None),
            prefix_cache_max_bytes=getattr(cfg, "prefix_cache_max_bytes", None),
            clear_cache_boundaries=getattr(cfg, "clear_cache_boundaries", None),
            max_snapshot_tokens=getattr(cfg, "max_snapshot_tokens", None),
            prefix_cache_l2=getattr(cfg, "prefix_cache_l2", None),
            prefix_cache_l2_dir=getattr(cfg, "prefix_cache_l2_dir", ""),
            prefix_cache_l2_max_bytes=getattr(cfg, "prefix_cache_l2_max_bytes", None),
            target_fa_window=getattr(cfg, "target_fa_window", 0) or 0,
            dflash_max_ctx=getattr(cfg, "dflash_max_ctx", 0) or 0,
            memory_waterfall=bool(getattr(cfg, "memory_waterfall", False)),
            bench_log_dir=getattr(cfg, "bench_log_dir", "") or "",
            verify_mode=getattr(cfg, "verify_mode", None),
        )

        diagnostics_mode = getattr(cfg, "diagnostics", "off") or "off"
        diagnostics_dir = getattr(cfg, "diagnostics_dir", None)
        bench_log_dir = getattr(cfg, "bench_log_dir", None)
        trace_dir = Path(diagnostics_dir) if diagnostics_dir else (Path(bench_log_dir) if bench_log_dir else None)
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_config = DiagnosticsConfig(
            mode=diagnostics_mode,
            run_dir=Path(diagnostics_dir) if diagnostics_dir else None,
            memory_waterfall=bool(getattr(cfg, "memory_waterfall", False) or diagnostics_mode == "full"),
            trace=TraceConfig(log_dir=trace_dir, cycle_events=diagnostics_mode == "full"),
        )

        self._runtime_context = build_runtime_context(runtime_config, diagnostics_config=diagnostics_config)
        wired_limit = getattr(cfg, "wired_limit", "auto") or "auto"
        cache_limit = getattr(cfg, "cache_limit", "auto") or "auto"
        metal_limits = apply_metal_limits(
            wired_request=parse_memory_limit(wired_limit) if isinstance(wired_limit, str) else wired_limit,
            cache_request=parse_memory_limit(cache_limit) if isinstance(cache_limit, str) else cache_limit,
        )
        self._runtime_context = with_metal_limits(self._runtime_context, metal_limits)

        target_model, tokenizer, draft_model, _resolved = load_runtime_components(
            model_ref=self._model_name,
            draft_ref=self._drafter_path,
            draft_quant=getattr(cfg, "draft_quant", None),
            verify_config=self._runtime_context.verify,
        )

        self._target_model = target_model
        self._draft_model = draft_model
        self._tokenizer = tokenizer
        self._model = target_model
        self._loaded = True
        self._dflash_ready = True

        rc = self._runtime_context.runtime
        logger.info(
            "[DFlash-MLX] Config: profile=%s verify=%s draft_sink=%s draft_window=%s "
            "prefix_cache=%s L1=%sx%s L2=%s max_snapshot=%s prefill_step=%s",
            getattr(rc, "profile", "balanced"), getattr(rc, "verify_mode", "auto"),
            getattr(rc, "draft_sink_size", 64), getattr(rc, "draft_window_size", 1024),
            getattr(rc, "prefix_cache", False),
            getattr(rc, "prefix_cache_max_entries", 4), getattr(rc, "prefix_cache_max_bytes", 0),
            getattr(rc, "prefix_cache_l2", False), getattr(rc, "max_snapshot_tokens", 24000),
            getattr(rc, "prefill_step_size", 4096),
        )
        gc.collect()
        logger.info("[DFlash-MLX] Models loaded, speculative decoding ready.")

    # ------------------------------------------------------------------ properties
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

    # ------------------------------------------------------------------ chat template
    def build_prompt(self, messages, tools=None, enable_thinking=None):
        from ..api.tool_calling import convert_tools_for_template
        template_tools = convert_tools_for_template(tools) if tools else None
        return self._apply_chat_template(messages, template_tools, enable_thinking=enable_thinking)

    # ------------------------------------------------------------------ generation
    async def generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9,
                       stop=None, images=None, videos=None, **kwargs):
        output_text = ""
        completion_tokens = 0
        finish_reason = "stop"
        async for chunk in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop=stop, images=images, videos=videos, **kwargs,
        ):
            output_text = chunk.text
            completion_tokens = chunk.completion_tokens
            finish_reason = chunk.finish_reason or "stop"
        prompt_tokens = self._count_prompt_tokens(prompt)
        return GenerationOutput(
            text=output_text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, finish_reason=finish_reason,
        )

    async def stream_generate(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9,
                              stop=None, images=None, videos=None, **kwargs):
        if not self._dflash_ready:
            raise RuntimeError("DFlash-MLX engine not loaded")

        tokenizer = self._tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        prompt_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        stop_token_ids: set[int] = set()
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)
        for s in (stop or []):
            stop_token_ids.update(tokenizer.encode(s))

        prefix_flow = self._build_prefix_flow(tokenizer, prompt_ids)

        from dflash_mlx.runtime import stream_dflash_generate

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _worker() -> None:
            try:
                handler = prefix_flow.get("handler")
                for event in stream_dflash_generate(
                    target_model=self._target_model, tokenizer=self._tokenizer,
                    draft_model=self._draft_model, prompt="",
                    max_new_tokens=max_tokens, use_chat_template=False,
                    stop_token_ids=stop_token_ids,
                    prompt_tokens_override=prompt_ids,
                    prefix_snapshot=prefix_flow.get("snapshot"),
                    stable_prefix_len=prefix_flow.get("stable_prefix_len"),
                    prefix_cache=prefix_flow.get("cache"),
                    runtime_context=self._runtime_context,
                ):
                    ename = event.get("event", "")
                    if ename == "prefill_snapshot_ready" and handler:
                        handler.handle_prefill_snapshot(event)
                    elif ename == "generation_snapshot_ready" and handler:
                        handler.handle_generation_snapshot(event)
                    loop.call_soon_threadsafe(q.put_nowait, event)
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, {"__error__": exc})

        self._model_load_executor.submit(_worker)

        full_text = ""
        completion_tokens = 0
        finish_reason = "stop"
        _decoder = self._tokenizer
        if hasattr(_decoder, "tokenizer"):
            _decoder = _decoder.tokenizer

        dflash_accepted = 0
        dflash_cycles = 0
        dflash_elapsed_us = 0
        dflash_prefill_us = 0
        dflash_draft_us = 0
        dflash_verify_us = 0
        dflash_acceptance = 0.0

        while True:
            event = await q.get()
            if event is None:
                break
            if isinstance(event, dict) and "__error__" in event:
                raise event["__error__"]

            ename = event.get("event", "")
            if ename == "token":
                token_id = int(event.get("token_id", 0))
                token_text = _decoder.decode([token_id])
                full_text += token_text
                completion_tokens = int(event.get("generated_tokens", completion_tokens + 1))
                if token_id in stop_token_ids:
                    finish_reason = "stop"
                yield GenerationOutput(
                    text=full_text, new_text=token_text,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    finished=False, finish_reason=None,
                )
            elif ename == "summary":
                completion_tokens = int(event.get("generation_tokens", completion_tokens))
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

        self._total_requests += 1
        self._total_tokens += completion_tokens
        self._total_accepted += dflash_accepted
        if prefix_flow.get("hit_tokens", 0) > 0:
            self._cache_hits += 1
            self._cache_tokens_saved += prefix_flow.get("hit_tokens", 0)
        else:
            self._cache_misses += 1

        cache_tag = "HIT({}/{})".format(prefix_flow.get("hit_tokens", 0), prompt_tokens) \
            if prefix_flow.get("hit_tokens", 0) > 0 else "MISS"
        tps = completion_tokens / (dflash_elapsed_us / 1_000_000) if dflash_elapsed_us > 0 else 0
        logger.info(
            "[DFlash-MLX] request: %d tokens  %.1f tok/s  accept=%.1f%%  "
            "accepted=%d/%d  cycles=%d  prefill=%.0fms  draft=%.0fms  verify=%.0fms  "
            "cache=%s  lookup=%.2fms",
            completion_tokens, tps, dflash_acceptance * 100,
            dflash_accepted, completion_tokens, dflash_cycles,
            dflash_prefill_us / 1000, dflash_draft_us / 1000, dflash_verify_us / 1000,
            cache_tag, prefix_flow.get("lookup_ms", 0),
        )

        yield GenerationOutput(
            text=full_text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, finished=True,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------ prefix cache
    def _build_prefix_flow(self, tokenizer, prompt_ids):
        try:
            from dflash_mlx.server.prefix_cache_flow import (
                PrefixCacheFlow, compute_stable_prefix_len, get_dflash_prefix_cache,
            )
            from dflash_mlx.server.prefix_cache_manager import (
                build_prefix_key, chat_template_marker_ids,
            )

            cache = get_dflash_prefix_cache(self._runtime_context)
            if cache is None:
                return {"cache": None, "snapshot": None, "stable_prefix_len": None,
                        "hit_tokens": 0, "lookup_ms": 0, "handler": None}

            class _StubProvider:
                model_key = (self._model_name, None, self._drafter_path)

            key = build_prefix_key(_StubProvider(), self._draft_model, self._runtime_context)
            im_start_id, assistant_id = chat_template_marker_ids(tokenizer)
            stable_prefix_len = compute_stable_prefix_len(
                prompt_ids, im_start_id=im_start_id, assistant_id=assistant_id,
            )

            lookup_tokens = prompt_ids[:stable_prefix_len]
            # Debug: compare against existing cache entries
            for eid, snap in sorted(getattr(cache, "_entries", {}).items(), key=lambda x: len(x[1].token_ids)):
                st = snap.token_ids
                common = 0
                for i in range(min(len(lookup_tokens), len(st))):
                    if lookup_tokens[i] != st[i]:
                        break
                    common += 1
                logger.info(
                    "[DFlash-MLX] cache entry %s: stored=%d lookup=%d "
                    "common=%d kind=%s first_4=%s",
                    str(eid)[:8], len(st), len(lookup_tokens),
                    common, getattr(snap, "kind", "?"), list(st[:4]),
                )
            lookup_t0 = time.perf_counter_ns()
            matched_len, snapshot = cache.lookup(lookup_tokens, key)
            lookup_ms = (time.perf_counter_ns() - lookup_t0) / 1e6
            hit_tokens = int(matched_len)

            c_entries = len(getattr(cache, "_entries", {}))
            c_stats = getattr(cache, "_stats", {})
            if hit_tokens > 0:
                logger.info(
                    "[DFlash-MLX] prefix cache HIT %d/%d tokens "
                    "(stable_prefix=%d, entries=%d, hits=%d, misses=%d, lookup=%.2fms)",
                    hit_tokens, len(prompt_ids), stable_prefix_len, c_entries,
                    c_stats.get("exact_hits", 0) + c_stats.get("prefix_hits", 0),
                    c_stats.get("misses", 0), lookup_ms,
                )
            else:
                logger.info(
                    "[DFlash-MLX] prefix cache MISS "
                    "(prompt=%d, stable_prefix=%d, entries=%d, "
                    "hits=%d, misses=%d, fingerprint_rejects=%d, lookup=%.2fms)",
                    len(prompt_ids), stable_prefix_len, c_entries,
                    c_stats.get("exact_hits", 0) + c_stats.get("prefix_hits", 0),
                    c_stats.get("misses", 0), c_stats.get("fingerprint_rejects", 0),
                    lookup_ms,
                )

            flow = PrefixCacheFlow(
                cache=cache, key=key, stable_prefix_len=stable_prefix_len,
                snapshot=snapshot, lookup_ms=lookup_ms, hit_tokens=hit_tokens,
                draft_model=self._draft_model, runtime_context=self._runtime_context,
            )
            return {
                "cache": cache, "snapshot": snapshot,
                "stable_prefix_len": stable_prefix_len,
                "hit_tokens": hit_tokens, "lookup_ms": lookup_ms, "handler": flow,
            }
        except Exception as exc:
            logger.debug("[DFlash-MLX] prefix cache init failed (non-fatal): %s", exc)
            return {"cache": None, "snapshot": None, "stable_prefix_len": None,
                    "hit_tokens": 0, "lookup_ms": 0, "handler": None}

    # ------------------------------------------------------------------ helpers
    def _count_prompt_tokens(self, prompt):
        t = self._tokenizer
        if hasattr(t, "tokenizer"):
            t = t.tokenizer
        return len(t.encode(prompt))

    # ------------------------------------------------------------------ stats
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
            base["dflash_avg_acceptance"] = self._total_accepted / max(self._total_tokens, 1)
            base["dflash_cache_hit_rate"] = self._cache_hits / self._total_requests
        return base
