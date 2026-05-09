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
        # Create our own executor for model loading — avoid using
        # BatchedEngine's scheduler executor which may not be ready yet.
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

        gc.collect()
        logger.info("[DFlash-MLX] Models loaded, ready.")

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
            finish_reason = chunk.finish_reason

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
        """Streaming generation via dflash-mlx."""
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
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
            stop_token_ids.add(tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 0)
        for s in stop_strs:
            ids = tokenizer.encode(s)
            stop_token_ids.update(ids)

        from dflash_mlx.runtime import stream_dflash_generate

        loop = asyncio.get_running_loop()
        gen_done = loop.create_future()

        def _run_gen() -> None:
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
                    runtime_context=self._runtime_context,
                )
                # Collect all output text
                events = list(event_iter)
                loop.call_soon_threadsafe(gen_done.set_result, events)
            except Exception as exc:
                loop.call_soon_threadsafe(gen_done.set_exception, exc)

        executor = self._model_load_executor
        executor.submit(_run_gen)

        events = await gen_done

        # Parse events into GenerationOutput chunks.
        # dflash-mlx token events carry token_id, not text — we decode here.
        full_text = ""
        completion_tokens = 0
        finish_reason = "stop"
        last_text = ""
        detokenizer = tokenizer.detokenizer if hasattr(tokenizer, "detokenizer") else tokenizer
        if hasattr(detokenizer, "reset"):
            detokenizer.reset()

        for event in events:
            event_name = event.get("event", "")
            if event_name == "token":
                token_id = int(event.get("token_id", 0))
                # Decode this single token to text
                try:
                    token_text = detokenizer.decode([token_id])
                except Exception:
                    token_text = tokenizer.decode([token_id]) if hasattr(tokenizer, "decode") else ""
                full_text += token_text
                completion_tokens = int(event.get("generated_tokens", completion_tokens + 1))

                # Check stop tokens
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
                last_text = full_text

            elif event_name == "summary":
                completion_tokens = int(event.get("tokens_generated", completion_tokens))
                finish_reason = event.get("finish_reason", "stop")

            elif event_name in ("prefill", "prefill_progress", "prefill_snapshot_ready", "generation_snapshot_ready"):
                pass

            elif event_name in ("cycle_complete", "memory_waterfall"):
                pass

        yield GenerationOutput(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finished=True,
            finish_reason=finish_reason,
        )

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Count tokens in a prompt string."""
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return len(tokenizer.encode(prompt))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        base = super().get_stats()
        base["engine_type"] = "dflash-mlx"
        base["dflash_ready"] = self._dflash_ready
        base["drafter_path"] = self._drafter_path
        return base
