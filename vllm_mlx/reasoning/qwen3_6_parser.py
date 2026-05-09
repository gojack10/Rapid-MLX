# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Qwen3.6 models — uses <thinking>...</thinking> tags."""

from .think_parser import BaseThinkingReasoningParser


class Qwen3_6ReasoningParser(BaseThinkingReasoningParser):
    """Parser for Qwen3.6's <thinking>...</thinking> reasoning markers."""

    @property
    def start_token(self) -> str:
        return "<thinking>"

    @property
    def end_token(self) -> str:
        return "</thinking>"
