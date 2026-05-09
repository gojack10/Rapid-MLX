# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-flight memory check (issue #324).

On low-memory Apple Silicon (e.g. Mac mini M4 24 GB), loading a model that
forces unified memory past ~85% of total can trip the iBoot AMCC async-abort
firmware path and **kernel-panic the entire machine** rather than raise a
userspace OOM. ``_check_memory_capacity`` warns the user before this
happens. It is best-effort — never aborts, falls through silently when it
can't read sizes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm_mlx.cli import _check_memory_capacity


def _fake_psutil(total_gb: float):
    fake = MagicMock()
    fake.virtual_memory.return_value = MagicMock(total=int(total_gb * (1024**3)))
    return fake


def _patch_size_bytes(monkeypatch, size_gb: float):
    """Stub the local-path branch of ``_check_memory_capacity`` to report
    a fixed model size without doing real I/O or HF lookups."""

    def _fake_isdir(p):
        return True

    def _fake_walk(p):
        # One file of the requested size at the model root.
        yield p, [], ["weights.safetensors"]

    def _fake_getsize(p):
        return int(size_gb * (1024**3))

    monkeypatch.setattr("os.path.isdir", _fake_isdir)
    monkeypatch.setattr("os.walk", _fake_walk)
    monkeypatch.setattr("os.path.getsize", _fake_getsize)


def test_warning_fires_on_24gb_mac_with_14gb_model(monkeypatch, capsys):
    """The exact issue #324 scenario: 14 GB Gemma-4-26B-4bit on a 24 GB
    Mac mini M4. Working set = 14 * 1.4 = 19.6 GB → 82% utilization → soft
    warning. The user gets a heads-up before MLX starts allocating."""
    _patch_size_bytes(monkeypatch, size_gb=14.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0)}):
        _check_memory_capacity("/local/path/to/gemma-4-26b")
    out = capsys.readouterr().out
    assert "Memory pressure" in out, f"expected warning, got: {out!r}"
    # 14 / 24 ≈ 0.58 raw, working set 19.6 / 24 ≈ 0.82 — soft tier.
    assert "82%" in out


def test_hard_warning_fires_on_kernel_panic_class(monkeypatch, capsys):
    """At ratio ≥ 0.85, the warning escalates to red + names the kernel
    panic risk + suggests --gpu-memory-utilization 0.75. Pin the message
    surface so a future refactor doesn't silently weaken it."""
    # 18 GB on 24 GB Mac → working set 25.2 GB → 105% utilization (catastrophic).
    _patch_size_bytes(monkeypatch, size_gb=18.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0)}):
        _check_memory_capacity("/local/path/to/large")
    out = capsys.readouterr().out
    assert "kernel panic" in out, f"hard warning must name the risk: {out!r}"
    assert "issue #324" in out
    assert "--gpu-memory-utilization 0.75" in out


def test_no_warning_with_comfortable_headroom(monkeypatch, capsys):
    """A small model on a big machine must produce zero output. The check
    is best-effort and should be invisible when it has nothing to add."""
    # 4 GB model on 96 GB Mac Studio → working set 5.6 GB → 5.8% utilization.
    _patch_size_bytes(monkeypatch, size_gb=4.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(96.0)}):
        _check_memory_capacity("/local/path/to/small")
    out = capsys.readouterr().out
    assert out == "", f"comfortable model must not warn; got: {out!r}"


def test_silent_when_psutil_unavailable(monkeypatch, capsys):
    """Best-effort: if psutil can't be imported, fall through silently
    rather than blocking startup."""
    _patch_size_bytes(monkeypatch, size_gb=20.0)

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _no_psutil(name, *a, **kw):
        if name == "psutil":
            raise ImportError("psutil not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _no_psutil)
    _check_memory_capacity("/local/path/to/anything")
    out = capsys.readouterr().out
    assert out == "", f"missing psutil must be silent; got: {out!r}"


def test_silent_when_size_lookup_fails(monkeypatch, capsys):
    """If neither local path nor HF cache nor HF API can resolve the size,
    skip the check — the loader's error paths handle real failures."""

    def _fake_isdir(p):
        return False

    monkeypatch.setattr("os.path.isdir", _fake_isdir)

    # HF lookups all return 0 / raise.
    def _no_cache(*a, **kw):
        return None

    def _api_fail(*a, **kw):
        raise RuntimeError("offline")

    with (
        patch("huggingface_hub.try_to_load_from_cache", _no_cache),
        patch("huggingface_hub.model_info", _api_fail),
        patch.dict("sys.modules", {"psutil": _fake_psutil(24.0)}),
    ):
        _check_memory_capacity("mlx-community/Some-Unreachable-Model")
    out = capsys.readouterr().out
    assert out == "", f"unresolvable size must be silent; got: {out!r}"


def test_never_calls_sys_exit(monkeypatch):
    """Defensive: the memory check is advisory, must never abort startup
    even on a catastrophic mismatch (212 GB model on 8 GB machine)."""
    _patch_size_bytes(monkeypatch, size_gb=212.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(8.0)}):
        # Must NOT raise SystemExit. capsys not asserted — we just need
        # the call to complete.
        _check_memory_capacity("/local/path/to/huge")


def test_warning_includes_actionable_recommendations(monkeypatch, capsys):
    """The warning must give the user a concrete next step (rapid-mlx
    models / --gpu-memory-utilization), not just describe the problem.
    Pins the actionability of the message."""
    _patch_size_bytes(monkeypatch, size_gb=14.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0)}):
        _check_memory_capacity("/local/path/to/gemma-4-26b")
    out = capsys.readouterr().out
    # Soft tier (82%) recommends just --gpu-memory-utilization 0.85.
    assert "--gpu-memory-utilization" in out
