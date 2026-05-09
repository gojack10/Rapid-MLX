# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Qwen3.6 — same <think>...</think> tags as Qwen3."""

from .think_parser import BaseThinkingReasoningParser


class Qwen3_6ReasoningParser(BaseThinkingReasoningParser):
    """Parser for Qwen3.6 — uses <think>...</think> (identical to Qwen3)."""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"
