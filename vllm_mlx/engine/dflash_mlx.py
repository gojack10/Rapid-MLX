# SPDX-License-Identifier: Apache-2.0
"""
DFlash-MLX speculative-decoding engine adapter for Rapid-MLX.
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
import threading
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
        self._snapshot_executor = None
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
        self._snapshot_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dflash-snapshot"
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
            draft_block_tokens=getattr(cfg, "draft_block_tokens", None),
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
            speculative_mode=getattr(cfg, "speculative_mode", None),
            ddtree_budget=getattr(cfg, "ddtree_budget", None),
            ddtree_topk=getattr(cfg, "ddtree_topk", None),
            ddtree_dense_mask=getattr(cfg, "ddtree_dense_mask", None),
            generation_snapshot=getattr(cfg, "generation_snapshot", None),
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
            "[DFlash-MLX] Config: profile=%s verify=%s speculative=%s ddtree_budget=%s ddtree_topk=%s "
            "draft_sink=%s draft_window=%s prefix_cache=%s L1=%sx%s L2=%s max_snapshot=%s "
            "prefill_step=%s gen_snapshot=%s repetition_penalty=%s",
            getattr(rc, "profile", "balanced"), getattr(rc, "verify_mode", "auto"),
            getattr(rc, "speculative_mode", "dflash"),
            getattr(rc, "ddtree_budget", None), getattr(rc, "ddtree_topk", None),
            getattr(rc, "draft_sink_size", 64), getattr(rc, "draft_window_size", 1024),
            getattr(rc, "prefix_cache", False),
            getattr(rc, "prefix_cache_max_entries", 4), getattr(rc, "prefix_cache_max_bytes", 0),
            getattr(rc, "prefix_cache_l2", False), getattr(rc, "max_snapshot_tokens", 24000),
            getattr(rc, "prefill_step_size", 4096), getattr(rc, "generation_snapshot", True),
            getattr(rc, "repetition_penalty", 1.0),
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
            cached_tokens=getattr(self, "_last_request_cached_tokens", 0),
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

        request_wall_start_ns = time.perf_counter_ns()
        token_decode_ns = 0
        event_process_ns = 0
        queue_wait_ns = 0
        worker_started = threading.Event()

        def _worker() -> None:
            nonlocal queue_wait_ns
            queue_wait_ns = time.perf_counter_ns() - request_wall_start_ns
            worker_started.set()
            try:
                handler = prefix_flow.get("handler")
                configured_block_tokens = int(getattr(self._runtime_context.runtime, "draft_block_tokens", 0) or 0)
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
                    block_tokens=(configured_block_tokens if configured_block_tokens > 0 else None),
                ):
                    ename = event.get("event", "")
                    if ename == "prefill_snapshot_ready" and handler:
                        handler.handle_prefill_snapshot(event)
                        continue
                    if ename == "generation_snapshot_ready" and handler:
                        executor = self._snapshot_executor or self._model_load_executor
                        executor.submit(handler.handle_generation_snapshot, event)
                        nonlocal_snapshot_count[0] += 1
                        continue
                    loop.call_soon_threadsafe(q.put_nowait, event)
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, {"__error__": exc})

        nonlocal_snapshot_count = [0]
        self._model_load_executor.submit(_worker)
        worker_started.wait()  # capture queue_wait_ns before entering async loop

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
        dflash_replay_us = 0
        dflash_commit_us = 0
        dflash_generation_snapshot_us = 0
        dflash_yield_pause_us = 0
        dflash_peak_memory_gb = None
        dflash_tokens_per_cycle = 0.0
        dflash_acceptance = 0.0
        dflash_post_prefill_wall_tps = 0.0
        dflash_post_prefill_core_tps = 0.0
        dflash_decode_to_last_wall_tps = 0.0
        dflash_decode_to_last_core_tps = 0.0
        dflash_cycle_wall_ms = 0.0
        dflash_cycle_core_ms = 0.0
        dflash_cycle_measured_avg_ms = 0.0
        dflash_decode_timings_us: dict[str, Any] = {}
        dflash_ddtree_summary: dict[str, Any] = {}
        dflash_ddtree_timing_avg_us: dict[str, Any] = {}
        dflash_ddtree_timing_totals_us: dict[str, Any] = {}
        dflash_prefetch: dict[str, Any] = {}
        prefill_event_wall_ns = 0
        first_token_wall_ns = 0
        last_token_wall_ns = 0
        q_get_wait_ns = 0
        token_yield_pause_ns = 0
        max_queue_depth = 0

        while True:
            _q_wait_start_ns = time.perf_counter_ns()
            event = await q.get()
            q_get_wait_ns += time.perf_counter_ns() - _q_wait_start_ns
            max_queue_depth = max(max_queue_depth, q.qsize())
            if event is None:
                break
            if isinstance(event, dict) and "__error__" in event:
                raise event["__error__"]

            event_process_start_ns = time.perf_counter_ns()
            ename = event.get("event", "")
            if ename == "token":
                token_id = int(event.get("token_id", 0))
                decode_start_ns = time.perf_counter_ns()
                token_text = _decoder.decode([token_id])
                token_decode_ns += time.perf_counter_ns() - decode_start_ns
                full_text += token_text
                completion_tokens = int(event.get("generated_tokens", completion_tokens + 1))
                if token_id in stop_token_ids:
                    finish_reason = "stop"
                    logger.info(
                        "[DFlash-MLX] EOS hit: token=%d text=%r at pos %d",
                        token_id, token_text, completion_tokens,
                    )
                # Log every 500th token for sampling
                if completion_tokens % 500 == 0:
                    logger.info(
                        "[DFlash-MLX] token %d: id=%d text=%r",
                        completion_tokens, token_id, token_text,
                    )
                now_ns = time.perf_counter_ns()
                if first_token_wall_ns == 0:
                    first_token_wall_ns = now_ns
                last_token_wall_ns = now_ns
                event_process_ns += now_ns - event_process_start_ns
                token_output = GenerationOutput(
                    text=full_text, new_text=token_text,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    cached_tokens=int(prefix_flow.get("hit_tokens", 0) or 0),
                    finished=False, finish_reason=None,
                )
                _yield_start_ns = time.perf_counter_ns()
                yield token_output
                token_yield_pause_ns += time.perf_counter_ns() - _yield_start_ns
                continue
            elif ename == "prefill":
                prefill_event_wall_ns = time.perf_counter_ns()
            elif ename == "cycle_complete":
                diagnostics_mode = getattr(
                    getattr(self._runtime_context, "diagnostics", None), "mode", "off"
                )
                if diagnostics_mode == "full":
                    cycle_fields = {k: v for k, v in event.items() if k != "event"}
                    logger.info(
                        "[DFlash-MLX] cycle_profile %s",
                        json.dumps(cycle_fields, separators=(",", ":"), default=str),
                    )
            elif ename == "summary":
                completion_tokens = int(event.get("generation_tokens", completion_tokens))
                finish_reason = event.get("finish_reason", "stop")
                logger.info(
                    "[DFlash-MLX] summary: tokens=%d finish=%s generated_ids_last10=%s",
                    completion_tokens, finish_reason,
                    event.get("generated_token_ids", [])[-10:] if event.get("generated_token_ids") else [],
                )
                ddtree_totals = event.get("ddtree_profile_totals_us", {})
                if ddtree_totals:
                    logger.info(
                        "[DFlash-MLX] ddtree_profile_us %s",
                        json.dumps(
                            {k: round(float(v), 1) for k, v in ddtree_totals.items()},
                            separators=(",", ":"),
                        ),
                    )
                dflash_accepted = int(event.get("accepted_from_draft", 0))
                dflash_cycles = int(event.get("cycles_completed", 0))
                dflash_elapsed_us = float(event.get("elapsed_us", 0))
                dflash_acceptance = float(event.get("acceptance_ratio", 0))
                dflash_tokens_per_cycle = float(event.get("tokens_per_cycle", 0) or 0)
                dflash_peak_memory_gb = event.get("peak_memory_gb")
                dflash_post_prefill_wall_tps = float(event.get("post_prefill_wall_tps", 0) or 0)
                dflash_post_prefill_core_tps = float(event.get("post_prefill_core_tps", 0) or 0)
                dflash_decode_to_last_wall_tps = float(event.get("decode_to_last_token_wall_tps", 0) or 0)
                dflash_decode_to_last_core_tps = float(event.get("decode_to_last_token_core_tps", 0) or 0)
                dflash_cycle_wall_ms = float(event.get("cycle_wall_ms", 0) or 0)
                dflash_cycle_core_ms = float(event.get("cycle_core_ms", 0) or 0)
                dflash_cycle_measured_avg_ms = float(event.get("cycle_measured_avg_ms", 0) or 0)
                dflash_decode_timings_us = dict(event.get("decode_timings_us", {}) or {})
                dflash_ddtree_summary = dict(event.get("ddtree", {}) or {})
                dflash_ddtree_timing_avg_us = dict(event.get("ddtree_timing_avg_us", {}) or {})
                dflash_ddtree_timing_totals_us = dict(event.get("ddtree_timing_totals_us", {}) or {})
                dflash_prefetch = dict(event.get("prefetch", {}) or {})
                phase = event.get("phase_timings_us", {})
                if isinstance(phase, dict):
                    dflash_prefill_us = float(phase.get("prefill", 0))
                    dflash_draft_us = float(phase.get("draft", 0))
                    dflash_verify_us = float(phase.get("verify", 0))
                    dflash_replay_us = float(phase.get("replay", 0))
                    dflash_commit_us = float(phase.get("commit", 0))
                    dflash_generation_snapshot_us = float(phase.get("generation_snapshot", 0))
                    dflash_yield_pause_us = float(phase.get("yield_pause", 0))
            event_process_ns += time.perf_counter_ns() - event_process_start_ns

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
        wall_us = (time.perf_counter_ns() - request_wall_start_ns) / 1_000.0
        core_tps = completion_tokens / (dflash_elapsed_us / 1_000_000) if dflash_elapsed_us > 0 else 0
        wall_tps = completion_tokens / (wall_us / 1_000_000) if wall_us > 0 else 0
        overhead_us = max(0.0, wall_us - dflash_elapsed_us)
        snapshot_insert_ms = float(getattr(prefix_flow.get("handler"), "insert_ms", 0.0) or 0.0)
        first_token_ms = (
            (first_token_wall_ns - request_wall_start_ns) / 1e6
            if first_token_wall_ns else 0.0
        )
        prefill_to_first_ms = (
            (first_token_wall_ns - prefill_event_wall_ns) / 1e6
            if first_token_wall_ns and prefill_event_wall_ns else 0.0
        )
        route_to_last_wall_tps = (
            completion_tokens / ((last_token_wall_ns - prefill_event_wall_ns) / 1e9)
            if last_token_wall_ns and prefill_event_wall_ns and last_token_wall_ns > prefill_event_wall_ns
            else 0.0
        )
        logger.info(
            "[DFlash-MLX] request: %d tokens  core=%.1f tok/s wall=%.1f tok/s  "
            "accept=%.1f%%  accepted=%d/%d  cycles=%d  prefill=%.0fms  "
            "draft=%.0fms  verify=%.0fms  replay=%.0fms  commit=%.0fms  "
            "gen_snap=%.0fms  yield_pause=%.0fms  overhead=%.0fms  queue=%.0fms  "
            "decode=%.1fms  event_proc=%.1fms  snap_async=%d insert=%.1fms  tpc=%.2f  "
            "peak=%.1fGB  cache=%s  lookup=%.2fms  finish=%s",
            completion_tokens, core_tps, wall_tps,
            dflash_acceptance * 100,
            dflash_accepted, completion_tokens, dflash_cycles,
            dflash_prefill_us / 1000, dflash_draft_us / 1000, dflash_verify_us / 1000,
            dflash_replay_us / 1000, dflash_commit_us / 1000,
            dflash_generation_snapshot_us / 1000, dflash_yield_pause_us / 1000,
            overhead_us / 1000, queue_wait_ns / 1e6,
            token_decode_ns / 1e6, event_process_ns / 1e6,
            nonlocal_snapshot_count[0], snapshot_insert_ms,
            dflash_tokens_per_cycle,
            float(dflash_peak_memory_gb) if dflash_peak_memory_gb is not None else 0.0,
            cache_tag, prefix_flow.get("lookup_ms", 0),
            finish_reason,
        )
        logger.info(
            "[DFlash-MLX] decode_observe: post_prefill wall=%.1f core=%.1f tok/s  "
            "to_last wall=%.1f core=%.1f tok/s  route_to_last=%.1f tok/s  "
            "cycle wall=%.2fms core=%.2fms measured=%.2fms  first=%.1fms prefill_to_first=%.1fms  "
            "q_wait=%.1fms token_yield_pause=%.1fms max_q=%d decode_timings_us=%s prefetch=%s",
            dflash_post_prefill_wall_tps, dflash_post_prefill_core_tps,
            dflash_decode_to_last_wall_tps, dflash_decode_to_last_core_tps,
            route_to_last_wall_tps,
            dflash_cycle_wall_ms, dflash_cycle_core_ms, dflash_cycle_measured_avg_ms,
            first_token_ms, prefill_to_first_ms,
            q_get_wait_ns / 1e6, token_yield_pause_ns / 1e6, max_queue_depth,
            json.dumps({k: round(float(v), 1) for k, v in dflash_decode_timings_us.items()}, separators=(",", ":")),
            json.dumps(dflash_prefetch, separators=(",", ":"), default=str),
        )
        if dflash_ddtree_timing_avg_us or dflash_ddtree_summary:
            logger.info(
                "[DFlash-MLX] ddtree_observe: summary=%s avg_us=%s totals_ms=%s",
                json.dumps(
                    {k: round(float(v), 3) if isinstance(v, (int, float)) else v for k, v in dflash_ddtree_summary.items()},
                    separators=(",", ":"),
                    default=str,
                ),
                json.dumps(
                    {k: round(float(v), 1) for k, v in dflash_ddtree_timing_avg_us.items()},
                    separators=(",", ":"),
                ),
                json.dumps(
                    {k: round(float(v) / 1000.0, 2) for k, v in dflash_ddtree_timing_totals_us.items()},
                    separators=(",", ":"),
                ),
            )

        self._last_request_cached_tokens = int(prefix_flow.get("hit_tokens", 0) or 0)
        yield GenerationOutput(
            text=full_text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=self._last_request_cached_tokens,
            finished=True,
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
            # Debug: compare against existing cache entries and remember the
            # longest divergent match so misses tell us exactly what moved.
            best_divergent: tuple[Any, Any, int] | None = None
            for eid, snap in sorted(getattr(cache, "_entries", {}).items(), key=lambda x: len(x[1].token_ids)):
                st = snap.token_ids
                common = 0
                for i in range(min(len(lookup_tokens), len(st))):
                    if lookup_tokens[i] != st[i]:
                        break
                    common += 1
                if common > 0 and common < min(len(lookup_tokens), len(st)):
                    if best_divergent is None or common > best_divergent[2]:
                        best_divergent = (eid, snap, common)
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
            if hit_tokens == 0:
                # Full prefix lookup missed. Scan for checkpoint snapshots —
                # prefill snapshots captured at safe chunk boundaries that
                # match a prefix of the current prompt. Checkpoints are stored
                # L2-only (SSD) to save GPU memory, so we search L2 directly.
                best_checkpoint_len = 0
                best_checkpoint_snap = None
                best_checkpoint_id = -1
                # First check L1 entries (hot cache, may still have checkpoints
                # from before the L2-only change or as fallback when L2 is absent).
                for eid, snap in cache._entries.items():
                    if snap.key != key:
                        continue
                    if snap.kind != "prefill":
                        continue
                    snap_len = len(snap.token_ids)
                    if snap_len == 0 or snap_len > len(lookup_tokens):
                        continue
                    if tuple(lookup_tokens[:snap_len]) == snap.token_ids:
                        if snap_len > best_checkpoint_len:
                            best_checkpoint_id = eid
                            best_checkpoint_len = snap_len
                            best_checkpoint_snap = snap
                # If no checkpoint in L1, try L2 (primary store for checkpoints).
                if best_checkpoint_snap is None and hasattr(cache, '_l2') and cache._l2 is not None:
                    l2_snap = cache._l2.lookup(lookup_tokens, key)
                    if l2_snap is not None:
                        l2_len = len(l2_snap.token_ids)
                        if l2_len > best_checkpoint_len:
                            best_checkpoint_len = l2_len
                            best_checkpoint_snap = l2_snap
                if best_checkpoint_len > 0:
                    snapshot = best_checkpoint_snap
                    hit_tokens = best_checkpoint_len
                    # Promote the checkpoint in LRU order so it survives
                    # eviction during this request's prefill checkpoint inserts.
                    if best_checkpoint_id >= 0:
                        try:
                            if best_checkpoint_id in cache._lru_order:
                                cache._lru_order.remove(best_checkpoint_id)
                                cache._lru_order.append(best_checkpoint_id)
                        except Exception:
                            pass
                    source = "L1" if best_checkpoint_id >= 0 else "L2"
                    logger.info(
                        "[DFlash-MLX] checkpoint HIT %d/%d tokens "
                        "(source=%s, entries=%d, will prefill tail %d..%d, lookup_ms=%.2fms)",
                        best_checkpoint_len, len(prompt_ids),
                        source, c_entries,
                        best_checkpoint_len, len(prompt_ids), lookup_ms,
                    )
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
                diagnostics_mode = getattr(getattr(self._runtime_context, "diagnostics", None), "mode", "off")
                if diagnostics_mode != "off" and best_divergent is not None:
                    eid, snap, common = best_divergent
                    self._log_prefix_divergence(tokenizer, lookup_tokens, snap, common, eid)

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
    def _decode_debug_tokens(self, tokenizer, token_ids) -> str:
        try:
            dec = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            text = dec.decode(list(token_ids))
        except Exception as exc:
            return f"<decode failed: {exc}>"
        text = text.replace("\n", "\\n")
        if len(text) > 1400:
            text = text[:1400] + "…"
        return text

    def _log_prefix_divergence(self, tokenizer, lookup_tokens, snap, common: int, eid: Any) -> None:
        cached_tokens = snap.token_ids
        window = 50  # ~100 tokens centered on the first mismatch.
        start = max(0, common - window)
        lookup_end = min(len(lookup_tokens), common + window)
        cached_end = min(len(cached_tokens), common + window)
        lookup_next = lookup_tokens[common] if common < len(lookup_tokens) else None
        cached_next = cached_tokens[common] if common < len(cached_tokens) else None
        logger.info(
            "[DFlash-MLX] prefix divergence best_entry=%s kind=%s stored=%d "
            "lookup=%d common=%d lookup_next=%s cached_next=%s",
            str(eid)[:8], getattr(snap, "kind", "?"), len(cached_tokens),
            len(lookup_tokens), common, lookup_next, cached_next,
        )
        logger.info(
            "[DFlash-MLX] lookup text around divergence [%d:%d]: %s",
            start, lookup_end,
            self._decode_debug_tokens(tokenizer, lookup_tokens[start:lookup_end]),
        )
        logger.info(
            "[DFlash-MLX] cached text around divergence [%d:%d]: %s",
            start, cached_end,
            self._decode_debug_tokens(tokenizer, cached_tokens[start:cached_end]),
        )
        probe = 20_000
        if common >= probe and len(lookup_tokens) > probe and len(cached_tokens) > probe:
            probe_start = max(0, probe - window)
            probe_lookup_end = min(len(lookup_tokens), probe + window)
            probe_cached_end = min(len(cached_tokens), probe + window)
            logger.info(
                "[DFlash-MLX] lookup text around token~20K [%d:%d]: %s",
                probe_start, probe_lookup_end,
                self._decode_debug_tokens(tokenizer, lookup_tokens[probe_start:probe_lookup_end]),
            )
            logger.info(
                "[DFlash-MLX] cached text around token~20K [%d:%d]: %s",
                probe_start, probe_cached_end,
                self._decode_debug_tokens(tokenizer, cached_tokens[probe_start:probe_cached_end]),
            )

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
