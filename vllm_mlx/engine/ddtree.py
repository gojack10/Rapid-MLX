"""
DDTree speculative decoding via tree verifier — Rapid-MLX integration.

Pure Python/NumPy module for building optimal prefix-closed draft trees
from DFlash block diffusion logits, and walking verified trees.

Implements Algorithm 1 from the DDTree paper: best-first heap search over
per-position draft distributions under a fixed node budget.
"""

from __future__ import annotations

import heapq
from functools import lru_cache
from typing import NamedTuple

import numpy as np

# ──────────────────────────────────────────────────────────────── DDTree ─────

class DDTree(NamedTuple):
    """Result of tree construction.

    All indices are 0-based with index 0 = root (the bonus token).
    Nodes 1..N are the drafted tree nodes.
    """

    node_token_ids: np.ndarray  # (N,) int64 — token ID for each tree node
    node_depths: np.ndarray     # (N,) int64 — depth of each node (root=0, children=1..L)
    parents: list[int]          # (N+1,) — parent index for each node; parents[0] = -1 (root)
    child_maps: list[dict[int, int]]  # (N+1,) — {token_id: child_index} for each node
    visibility: np.ndarray      # (N+1, N+1) bool — ancestor-only attention mask
    node_count: int             # number of tree nodes (excluding root)

    @property
    def tree_size(self) -> int:
        """Total number of nodes including root."""
        return 1 + self.node_count


# ────────────────────────────────────────────────── build_ddtree_tree ────────

def build_ddtree_tree_from_topk(
    top_token_ids: np.ndarray,
    top_log_probs: np.ndarray,
    budget: int,
    min_cumulative_log_prob: float = float('-inf'),
    max_depth: int = 0,
    depth_penalty: float = 0.0,
) -> DDTree:
    """Build a DDTree from precomputed per-position top-k log-probs.

    Args:
        top_token_ids: (L, K) int64 array sorted by descending log-probability.
        top_log_probs: (L, K) float32 array aligned with top_token_ids.
        budget: Maximum number of tree nodes (excluding root).
        min_cumulative_log_prob: Prune paths whose cumulative log-probability
            falls below this threshold.  Default ``-inf`` (no pruning).
        max_depth: Maximum tree depth. 0 = unlimited. Caps the rank-0 chain
            length so budget is spent on branching at shallow depths.
        depth_penalty: Per-depth penalty added to child cumulative score.
            Makes deeper nodes slightly less attractive. 0.0 = no penalty.

    Returns:
        DDTree with up to *budget* tree nodes.
    """
    if budget <= 0 or top_token_ids.shape[0] == 0 or top_token_ids.shape[1] == 0:
        visibility = np.zeros((1, 1), dtype=np.bool_)
        visibility[0, 0] = True
        return DDTree(
            node_token_ids=np.empty(0, dtype=np.int64),
            node_depths=np.empty(0, dtype=np.int64),
            parents=[-1],
            child_maps=[{}],
            visibility=visibility,
            node_count=0,
        )

    top_token_ids = np.asarray(top_token_ids, dtype=np.int64)
    top_log_probs = np.asarray(top_log_probs, dtype=np.float32)
    topk = min(int(budget), int(top_token_ids.shape[1]))
    depth_limit = int(top_token_ids.shape[0])

    # Best-first heap search (Algorithm 1 from DDTree paper).
    # Heap entries: (-logw, ranks_tuple, parent_index, depth, rank, logw)
    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids = np.empty(budget, dtype=np.int64)
    node_depths = np.empty(budget, dtype=np.int64)
    parents = np.empty(budget + 1, dtype=np.int32)
    parents[0] = -1
    child_maps: list[dict[int, int]] = [{}]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents[current_index] = parent_index
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Push sibling (next rank at same depth)
        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs[depth - 1, rank])
                + float(top_log_probs[depth - 1, rank + 1])
            )
            if sibling_logw >= min_cumulative_log_prob:
                heapq.heappush(
                    heap,
                    (
                        -sibling_logw,
                        sibling_ranks,
                        parent_index,
                        depth,
                        rank + 1,
                        sibling_logw,
                    ),
                )

        # Push first child (rank 0 at next depth)
        if depth < depth_limit and (max_depth <= 0 or depth < max_depth):
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0]) + depth_penalty
            if child_logw >= min_cumulative_log_prob:
                heapq.heappush(
                    heap,
                    (
                        -child_logw,
                        child_ranks,
                        current_index,
                        depth + 1,
                        0,
                        child_logw,
                    ),
                )

    # Build visibility matrix (ancestor-only attention mask).
    # Node i can attend to node j iff j is an ancestor of i (or j == i).
    current_length = 1 + node_count
    visibility = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents[index])
        visibility[index, :index] = visibility[parent_index, :index]
        visibility[index, index] = True

    return DDTree(
        node_token_ids=node_token_ids[:node_count],
        node_depths=node_depths[:node_count],
        parents=parents[:current_length].tolist(),
        child_maps=child_maps,
        visibility=visibility,
        node_count=node_count,
    )


# ────────────────────────────────────────────── build_ddtree_tree ────────────

def build_ddtree_tree(
    draft_logits: np.ndarray,
    budget: int,
) -> DDTree:
    """Build an optimal draft tree from DFlash block diffusion logits.

    Args:
        draft_logits: (L, vocab_size) float32 — per-position logits from the
            DFlash draft model for positions 1..L after the bonus token.
        budget: Maximum number of tree nodes (excluding root).
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    topk = min(budget, draft_logits.shape[-1])
    logits = draft_logits.astype(np.float32)
    top_indices = np.argpartition(-logits, topk - 1, axis=-1)[:, :topk]
    top_logits = np.take_along_axis(logits, top_indices, axis=-1)
    sort_order = np.argsort(-top_logits, axis=-1)
    top_token_ids = np.take_along_axis(top_indices, sort_order, axis=-1)
    top_logits = np.take_along_axis(top_logits, sort_order, axis=-1)
    logits_max = logits.max(axis=-1, keepdims=True)
    log_z = (
        np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
        + logits_max
    )
    top_log_probs = top_logits - log_z
    return build_ddtree_tree_from_topk(
        top_token_ids=top_token_ids.astype(np.int64),
        top_log_probs=top_log_probs,
        budget=budget,
    )


def build_ddtree_tree_from_mlx_topk(
    top_token_ids: "mx.array",
    top_log_probs: "mx.array",
    budget: int,
    profile: dict | None = None,
    min_cumulative_log_prob: float = float('-inf'),
    max_depth: int = 0,
    depth_penalty: float = 0.0,
) -> DDTree:
    """Build a DDTree from MLX top-k token IDs/log-probs.

    Transfers only the compact (L, K) arrays to CPU for heap construction.
    When ``profile`` is provided, it is populated with synchronized phase
    timings that split draft-topk materialization from Python heap work.
    """
    import mlx.core as mx
    import time

    def _profile_add(name: str, elapsed_ns: int) -> None:
        if profile is not None:
            profile[name] = int(profile.get(name, 0)) + int(elapsed_ns)

    if budget <= 0 or int(top_token_ids.shape[0]) == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )
    _phase_start = time.perf_counter_ns() if profile is not None else 0
    top_token_ids = top_token_ids.astype(mx.uint32)
    top_log_probs = top_log_probs.astype(mx.float32)
    if profile is not None:
        _profile_add("topk_cast_ns", time.perf_counter_ns() - _phase_start)
        _phase_start = time.perf_counter_ns()
    mx.eval(top_token_ids, top_log_probs)
    if profile is not None:
        _profile_add("topk_sync_ns", time.perf_counter_ns() - _phase_start)
        _phase_start = time.perf_counter_ns()
    ids_np = np.array(top_token_ids.tolist(), dtype=np.int64)
    probs_np = np.array(top_log_probs.tolist(), dtype=np.float32)
    if profile is not None:
        _profile_add("topk_transfer_ns", time.perf_counter_ns() - _phase_start)
        _phase_start = time.perf_counter_ns()
    tree = build_ddtree_tree_from_topk(
        top_token_ids=ids_np,
        top_log_probs=probs_np,
        budget=budget,
        min_cumulative_log_prob=min_cumulative_log_prob,
        max_depth=max_depth,
        depth_penalty=depth_penalty,
    )
    if profile is not None:
        _profile_add("heap_build_ns", time.perf_counter_ns() - _phase_start)
    return tree


def build_ddtree_tree_from_mlx(
    draft_logits: "mx.array",
    budget: int,
) -> DDTree:
    """Build a DDTree from MLX draft logits without CPU round-trip.

    Does top-K extraction and log-softmax on GPU via MLX, then transfers
    only the (L, topk) token IDs and log-probs to CPU for tree construction.

    Args:
        draft_logits: (L, vocab_size) MLX array — per-position logits.
        budget: Maximum number of tree nodes (excluding root).
    """
    import mlx.core as mx

    if budget <= 0 or int(draft_logits.shape[0]) == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    topk = min(budget, int(draft_logits.shape[-1]))
    dlogits = draft_logits.astype(mx.float32)
    top_indices = mx.argpartition(-dlogits, kth=topk - 1, axis=-1)[:, :topk]
    top_logits = mx.take_along_axis(dlogits, top_indices, axis=-1)
    sort_order = mx.argsort(-top_logits, axis=-1)
    top_token_ids = mx.take_along_axis(top_indices, sort_order, axis=-1)
    top_logits = mx.take_along_axis(top_logits, sort_order, axis=-1)
    top_log_probs = top_logits - mx.logsumexp(dlogits, axis=-1, keepdims=True)
    top_token_ids = top_token_ids.astype(mx.uint32)
    top_log_probs = top_log_probs.astype(mx.float32)
    mx.eval(top_token_ids, top_log_probs)

    # Transfer to CPU numpy via Python lists (reliable, no dtype issues)
    ids_np = np.array(top_token_ids.tolist(), dtype=np.int64)
    probs_np = np.array(top_log_probs.tolist(), dtype=np.float32)
    return build_ddtree_tree_from_topk(
        top_token_ids=ids_np,
        top_log_probs=probs_np,
        budget=budget,
    )


# ────────────────────────────────────────────── follow_verified_tree ─────────

def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree following the target model's greedy tokens.

    Starting at the root (index 0), check if the target model's chosen token
    matches a child in the tree.  If so, accept and continue.  The walk stops
    at the first mismatch; that token becomes the bonus for the next round.

    Args:
        child_maps: Per-node {token_id: child_index} maps from DDTree.
        posterior_tokens: Greedy argmax token for each tree node
            (length = tree_size).

    Returns:
        (accepted_indices, bonus_token):
        - accepted_indices: indices of accepted nodes (starting with 0 = root).
        - bonus_token: first unmatched target token (bonus for next round).
    """
    accepted_indices = [0]
    current_index = 0
    next_token = posterior_tokens[current_index]

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = posterior_tokens[current_index]

    return accepted_indices, next_token


# ────────────────────────────────────────────── compute_dfs_order ────────────

def compute_dfs_order(tree: DDTree) -> tuple[list[int], list[int]]:
    """Compute DFS traversal order for tree nodes.

    The heap construction already produces nodes roughly in probability order,
    but we need explicit DFS ordering for linear layer processing.

    Args:
        tree: DDTree from build_ddtree_tree.

    Returns:
        (dfs_order, inv_dfs_order):
        - dfs_order[i] = tree index at position i in DFS traversal.
        - inv_dfs_order[tree_index] = position in DFS.
    """
    if tree.node_count == 0:
        return [0], [0]

    n = 1 + tree.node_count
    children: list[list[int]] = [[] for _ in range(n)]
    for idx in range(1, n):
        parent = tree.parents[idx]
        children[parent].append(idx)

    dfs_order: list[int] = []
    stack = [0]
    while stack:
        node = stack.pop()
        dfs_order.append(node)
        for child in reversed(children[node]):
            stack.append(child)

    inv_dfs_order = [0] * n
    for pos, idx in enumerate(dfs_order):
        inv_dfs_order[idx] = pos

    return dfs_order, inv_dfs_order


# ────────────────────────────────────────────── CompiledTree ────────────────

class CompiledTree:
    """DDTree compiled into MLX tensors ready for verification."""

    __slots__ = (
        "input_ids", "position_ids", "attention_mask",
        "dfs_order", "inv_dfs_order",
        "parents", "depths", "tree_size",
    )

    def __init__(
        self,
        *,
        input_ids: "mx.array",
        position_ids: "mx.array",
        attention_mask: "mx.array",
        dfs_order: "mx.array",
        inv_dfs_order: "mx.array",
        parents: list[int],
        depths: list[int],
        tree_size: int,
    ):
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.dfs_order = dfs_order
        self.inv_dfs_order = inv_dfs_order
        self.parents = parents
        self.depths = depths
        self.tree_size = tree_size


def compile_tree(
    tree: DDTree,
    root_token_id: int,
    prefix_len: int,
) -> CompiledTree:
    """Compile a DDTree into MLX tensors for verification.

    Args:
        tree: DDTree from build_ddtree_tree.
        root_token_id: The bonus token (root of the tree).
        prefix_len: Number of tokens already in KV cache (context length).

    Returns:
        CompiledTree with all tensors needed for tree_verify_forward.
    """
    import mlx.core as mx

    tree_size = 1 + tree.node_count

    # 1. Input IDs: [root_token, node_0_token, node_1_token, ...]
    token_ids = np.empty(tree_size, dtype=np.int32)
    token_ids[0] = root_token_id
    if tree.node_count > 0:
        token_ids[1:] = tree.node_token_ids
    input_ids = mx.array(token_ids, dtype=mx.uint32)[None]

    # 2. Position IDs: root at prefix_len, each node at prefix_len + depth
    positions = np.empty(tree_size, dtype=np.int32)
    positions[0] = prefix_len
    if tree.node_count > 0:
        positions[1:] = prefix_len + tree.node_depths
    position_ids = mx.array(positions, dtype=mx.int32)
    depths: list[int] = [0]
    if tree.node_count > 0:
        depths.extend(int(depth) for depth in tree.node_depths.tolist())

    # 3. Tree-only attention mask (prefix attention handled by SDPA)
    mask = np.where(tree.visibility, 0.0, -np.inf).astype(np.float32)
    attention_mask = mx.array(mask)[None, None, :, :]

    # 4. DFS ordering for linear layers
    dfs, inv_dfs = compute_dfs_order(tree)
    dfs_order = mx.array(dfs, dtype=mx.int32)
    inv_dfs_order = mx.array(inv_dfs, dtype=mx.int32)

    return CompiledTree(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        dfs_order=dfs_order,
        inv_dfs_order=inv_dfs_order,
        parents=list(tree.parents),
        depths=depths,
        tree_size=tree_size,
    )


# ────────────────────────────────────────────── tree_verify_forward ──────────

def tree_verify_forward(
    target_model,
    *,
    compiled_tree: CompiledTree,
    cache: list,
    capture_layer_ids: set[int] | None = None,
    tree_cache_state: dict | None = None,
    compute_logits: bool = True,
    profile: dict | None = None,
) -> "tuple[mx.array | None, dict]":
    """Run the target model on all tree nodes with tree attention.

    Attention layers use tree visibility masks + per-token RoPE.
    Linear (GDN) layers use DFS-ordered sequential recurrence.

    Args:
        target_model: The loaded MLX target model.
        compiled_tree: CompiledTree from compile_tree().
        cache: List of per-layer caches.
        capture_layer_ids: Set of layer indices to capture hidden states.
        tree_cache_state: Optional dict for per-node state tracking.
        compute_logits: When False, skip the full tree LM head and store
            normalized hidden states in tree_cache_state.  Callers can then
            compute logits only for the accepted path.
        profile: Optional dict populated with synchronized verifier timing
            breakdowns. Intended only for diagnostics/full because it inserts
            synchronization points after setup, each layer, final norm, and the
            LM head.

    Returns:
        (logits, captured_hidden_states):
        - logits: (1, tree_size, vocab_size) in tree-index order, or None
          when compute_logits=False.
        - captured_hidden_states: {layer_id: (1, tree_size, hidden_dim)}.
    """
    import mlx.core as mx
    import time

    profile_enabled = profile is not None

    def _profile_add(name: str, elapsed_ns: int) -> None:
        if profile is not None:
            profile[name] = int(profile.get(name, 0)) + int(elapsed_ns)

    # Import dflash-mlx internals for model navigation
    from dflash_mlx.engine.target_ops import resolve_target_ops

    ct = compiled_tree
    inner = resolve_target_ops(target_model).text_model(target_model)

    # Determine actual KV cache prefix length from any attention cache
    actual_prefix: int = 0
    for c in cache:
        if hasattr(c, "offset") and not hasattr(c, "cache"):
            actual_prefix = int(getattr(c, "offset", 0) or 0)
            break
    if actual_prefix == 0:
        actual_prefix = ct.position_ids[0].item()

    dfs = ct.dfs_order
    inv_dfs = ct.inv_dfs_order

    # Embed tokens in tree-index order
    h = inner.embed_tokens(ct.input_ids)  # (1, tree_size, hidden_dim)

    # Build full attention mask: prefix (all attend) + tree visibility.
    # Attention KV for tree nodes is appended in DFS order, so both query rows
    # and tree-key columns must be DFS-reordered. Reordering only rows lets a
    # node attend to the wrong sibling/ancestor KV slot and breaks equivalence
    # with sequential verification.
    tree_vis = ct.attention_mask.astype(mx.float32)
    tree_vis_dfs = tree_vis[:, :, dfs, :][:, :, :, dfs]
    prefix_mask = mx.zeros((1, 1, ct.tree_size, actual_prefix), dtype=mx.float32)
    attention_mask_full_dfs = mx.concatenate([prefix_mask, tree_vis_dfs], axis=-1)

    def _tree_mask_for_key_len(key_len: int):
        """Build the DDTree mask for the physical KV length in this FA layer.

        RotatingKVCache keeps a logical offset that can be much larger than the
        physical KV tensor.  DDTree position ids must stay logical, but the SDPA
        mask must match the physical keys returned by update_and_fetch().
        """
        physical_prefix = max(0, int(key_len) - int(ct.tree_size))
        if physical_prefix == actual_prefix:
            return attention_mask_full_dfs
        prefix = mx.zeros((1, 1, ct.tree_size, physical_prefix), dtype=mx.float32)
        return mx.concatenate([prefix, tree_vis_dfs], axis=-1)

    if profile_enabled:
        _setup_sync_start = time.perf_counter_ns()
        mx.eval(h, attention_mask_full_dfs)
        _profile_add("setup_ns", time.perf_counter_ns() - _setup_sync_start)

    # For FA layers: reorder tokens to DFS, apply mask, reorder back
    position_ids_tree = ct.position_ids

    fa_idx_list = getattr(inner, "fa_idx", None)
    captured: dict[int, mx.array] = {}
    if capture_layer_ids and 0 in capture_layer_ids:
        captured[0] = h

    # For small budgets (B≤32): use flat DFS GDN (fast, rollback is cheap).
    # For large budgets (B>32): use tree-state GDN (no rollback cost).
    use_tree_state_gdn = True  # Tree-state GDN avoids corrupting GDN state during commit
    depth_groups = _group_by_depth(ct.parents, ct.depths) if use_tree_state_gdn else []

    if not use_tree_state_gdn:
        # Reorder to DFS for flat sequential GDN
        h = h[:, dfs, :]
        position_ids_dfs = ct.position_ids[dfs]
        tree_mask_dfs = attention_mask_full_dfs

    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        _layer_profile_start = time.perf_counter_ns() if profile_enabled else 0
        _layer_kind = "gdn" if getattr(layer, "is_linear", False) else "fa"
        if getattr(layer, "is_linear", False):
            if use_tree_state_gdn:
                # ── Tree-state GDN: per-node branching (large budgets) ──
                linear_input = layer.input_layernorm(h)
                store_state_all = tree_cache_state is None
                r, node_conv_states, node_states = _tree_state_gdn_forward(
                    layer.linear_attn, linear_input, layer_cache,
                    parents=ct.parents, depth_groups=depth_groups,
                    store_state_all=store_state_all,
                )
                if tree_cache_state is not None:
                    if node_states is None:
                        tree_cache_state.setdefault("gdn_recompute_layers", {})[layer_idx] = {
                            "linear_attn": layer.linear_attn,
                            "inputs": linear_input,
                        }
                    else:
                        tree_cache_state.setdefault("gdn_layers", {})[layer_idx] = {
                            "conv_states": node_conv_states,
                            "states": node_states,
                        }
            else:
                # ── Flat DFS sequential GDN (small budgets, fast) ──
                linear_input = layer.input_layernorm(h)
                r = layer.linear_attn(linear_input, None, layer_cache)
            h = h + r
            mlp_input = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(mlp_input)
            h = h + mlp_out
        else:
            # ── Attention (FA) layer: tree mask + per-token RoPE ──
            if use_tree_state_gdn:
                # h is in tree-index order; reorder to DFS for attention
                h_fa = h[:, dfs, :]
                pos_fa = position_ids_tree[dfs]
                mask_fa = attention_mask_full_dfs
            else:
                # h already in DFS order
                h_fa = h
                pos_fa = position_ids_dfs
                mask_fa = tree_mask_dfs

            attn = layer.self_attn
            attn_input = layer.input_layernorm(h_fa)
            B_val, L_val, D_val = attn_input.shape

            # Q projection + split into queries and gate
            q_proj_output = attn.q_proj(attn_input)
            queries, gate = mx.split(
                q_proj_output.reshape(B_val, L_val, attn.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B_val, L_val, -1)

            # K, V projections
            keys = attn.k_proj(attn_input)
            values = attn.v_proj(attn_input)

            # Reshape and normalize
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            keys = attn.k_norm(
                keys.reshape(B_val, L_val, attn.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                B_val, L_val, attn.num_key_value_heads, -1
            ).transpose(0, 2, 1, 3)

            # Per-token RoPE via batch-reshape trick
            _, H, T, D = queries.shape
            queries_rs = queries.transpose(0, 2, 1, 3).reshape(T, H, 1, D)
            queries_roped = attn.rope(queries_rs, offset=pos_fa)
            queries = queries_roped.reshape(1, T, H, D).transpose(0, 2, 1, 3)

            keys_rs = keys.transpose(0, 2, 1, 3).reshape(T, attn.num_key_value_heads, 1, D)
            keys_roped = attn.rope(keys_rs, offset=pos_fa)
            keys = keys_roped.reshape(1, T, attn.num_key_value_heads, D).transpose(0, 2, 1, 3)

            # Update KV cache
            if layer_cache is not None:
                keys, values = layer_cache.update_and_fetch(keys, values)

            mask_fa = _tree_mask_for_key_len(int(keys.shape[2]))

            # SDPA with tree+prefix mask.  For DDTree's small query counts,
            # MLX's native SDPA with the compact full mask is faster than the
            # Python-level split-prefix exact path even at long prefix lengths.
            # Keep the split fallback only for very large trees where a full
            # prefix-width mask may be too expensive.
            if actual_prefix >= 8192 and ct.tree_size > 512:
                output = _split_prefix_tree_attention(
                    queries=queries, keys=keys, values=values,
                    scale=attn.scale, tree_mask=tree_vis_dfs,
                    cached_prefix_len=actual_prefix,
                    repeat_kv=(attn.num_attention_heads != attn.num_key_value_heads),
                )
            else:
                output = mx.fast.scaled_dot_product_attention(
                    queries, keys, values, scale=attn.scale, mask=mask_fa.astype(queries.dtype)
                )
            output = output.transpose(0, 2, 1, 3).reshape(B_val, L_val, -1)
            r_fa = attn.o_proj(output * mx.sigmoid(gate))

            if use_tree_state_gdn:
                # Reorder back to tree-index order
                r = r_fa[:, inv_dfs, :]
            else:
                r = r_fa
                # h stays in DFS order for next linear layer
            h = h + r

            # MLP (operates on whatever order h is in)
            mlp_input = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(mlp_input)
            h = h + mlp_out

        if profile_enabled:
            _layer_sync_start = time.perf_counter_ns()
            mx.eval(h)
            _layer_elapsed = time.perf_counter_ns() - _layer_profile_start
            _layer_sync_elapsed = time.perf_counter_ns() - _layer_sync_start
            _profile_add(f"{_layer_kind}_layers_ns", _layer_elapsed)
            _profile_add(f"{_layer_kind}_layer_sync_ns", _layer_sync_elapsed)
            profile.setdefault("layers", []).append(
                {
                    "idx": int(layer_idx),
                    "kind": _layer_kind,
                    "us": _layer_elapsed / 1_000.0,
                }
            )

        if capture_layer_ids and (layer_idx + 1) in capture_layer_ids:
            if use_tree_state_gdn:
                captured[layer_idx + 1] = h  # tree-index order
            else:
                captured[layer_idx + 1] = h[:, inv_dfs, :]  # DFS→tree-index

    # Final norm and LM head
    _norm_profile_start = time.perf_counter_ns() if profile_enabled else 0
    if use_tree_state_gdn:
        normalized = inner.norm(h)  # h in tree-index
    else:
        normalized = inner.norm(h)
        normalized = normalized[:, inv_dfs, :]  # DFS→tree-index
    if profile_enabled:
        mx.eval(normalized)
        _profile_add("final_norm_ns", time.perf_counter_ns() - _norm_profile_start)

    _lm_head_profile_start = time.perf_counter_ns() if profile_enabled else 0
    logits = (
        resolve_target_ops(target_model).logits_from_hidden(target_model, normalized)
        if compute_logits
        else None
    )
    if profile_enabled and logits is not None:
        mx.eval(logits)
        _profile_add("lm_head_ns", time.perf_counter_ns() - _lm_head_profile_start)

    if tree_cache_state is not None:
        tree_cache_state["attention_append_order"] = "dfs"
        tree_cache_state["dfs_order"] = ct.dfs_order
        tree_cache_state["inv_dfs_order"] = ct.inv_dfs_order
        if not compute_logits:
            tree_cache_state["normalized_hidden"] = normalized

    return logits, captured


# ──────────────────────────── _split_prefix_tree_attention ───────────────────

def _split_prefix_tree_attention(
    *,
    queries: "mx.array",
    keys: "mx.array",
    values: "mx.array",
    scale: float,
    tree_mask: "mx.array",
    cached_prefix_len: int,
    repeat_kv: bool = True,
) -> "mx.array":
    """Exact attention over prefix + tree without a prefix-width mask.

    Splits keys/values into prefix (attends all) and tree (ancestor-only)
    portions, computing numerator+denominator separately for each, then
    combining via joint softmax.  Avoids materializing a (tree_size ×
    total_kv_len) mask.

    Returns output in ``queries.dtype``.
    """
    import mlx.core as mx

    query_heads = int(queries.shape[1])
    kv_heads = int(keys.shape[1])
    if repeat_kv and kv_heads != query_heads:
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"query heads ({query_heads}) not divisible by KV heads ({kv_heads})"
            )
        factor = query_heads // kv_heads
        keys = mx.repeat(keys, factor, axis=1)
        values = mx.repeat(values, factor, axis=1)

    prefix_len = int(cached_prefix_len)
    prefix_keys = keys[:, :, :prefix_len, :].astype(mx.float32)
    prefix_values = values[:, :, :prefix_len, :].astype(mx.float32)
    tree_keys = keys[:, :, prefix_len:, :].astype(mx.float32)
    tree_values = values[:, :, prefix_len:, :].astype(mx.float32)
    q = queries.astype(mx.float32)

    prefix_scores = mx.matmul(q, prefix_keys.transpose(0, 1, 3, 2)) * scale
    tree_scores = mx.matmul(q, tree_keys.transpose(0, 1, 3, 2)) * scale
    tree_scores = tree_scores + tree_mask.astype(mx.float32)

    prefix_max = mx.max(prefix_scores, axis=-1, keepdims=True)
    tree_max = mx.max(tree_scores, axis=-1, keepdims=True)
    joint_max = mx.maximum(prefix_max, tree_max)

    prefix_weights = mx.exp(prefix_scores - joint_max)
    tree_weights = mx.exp(tree_scores - joint_max)
    denom = (
        mx.sum(prefix_weights, axis=-1, keepdims=True)
        + mx.sum(tree_weights, axis=-1, keepdims=True)
    )
    output = (
        mx.matmul(prefix_weights, prefix_values)
        + mx.matmul(tree_weights, tree_values)
    ) / denom
    return output.astype(queries.dtype)


# ──────────────────── _linear_forward_tree_aware ─────────────────────────────

def _group_by_depth(parents: list[int], depths: list[int]) -> list[list[int]]:
    """Group tree node indices by depth (0 = root, 1 = children of root, ...).

    Returns list of lists where result[d] = indices of nodes at depth d.
    """
    groups: list[list[int]] = []
    for idx, depth in enumerate(depths):
        while len(groups) <= depth:
            groups.append([])
        groups[depth].append(idx)
    return groups


_TREE_GATED_DELTA_KERNEL = None
_TREE_GATED_DELTA_KERNEL_FAILED = False
_TREE_GATED_DELTA_NOSTATE_KERNEL = None
_TREE_GATED_DELTA_NOSTATE_KERNEL_FAILED = False


@lru_cache(maxsize=512)
def _conv_window_indices_cached(parents_tuple: tuple[int, ...], keep: int) -> np.ndarray:
    """Gather table for tree-branching depthwise conv windows.

    Entries index a table ``[base_conv_state, qkv_tree_nodes]``.  For each
    node, the row contains the parent's ``keep`` qkv history followed by the
    current node qkv, matching ``conv_input = concat(parent_state, qkv_step)``.
    """
    parents = list(parents_tuple)
    tree_size = len(parents)
    window = np.empty((tree_size, keep + 1), dtype=np.int32)

    for node_idx in range(tree_size):
        # Ancestors before the current node in chronological order.
        ancestors: list[int] = []
        cur = int(parents[node_idx])
        while cur >= 0:
            ancestors.append(cur)
            cur = int(parents[cur])
        ancestors.reverse()

        previous = ancestors[-keep:] if keep > 0 else []
        missing = keep - len(previous)
        row: list[int] = []
        if missing > 0:
            # Use the newest missing entries from the cached convolution state.
            row.extend(range(keep - missing, keep))
        row.extend(keep + int(idx) for idx in previous)
        row.append(keep + int(node_idx))
        window[node_idx, :] = row

    return window


@lru_cache(maxsize=512)
def _conv_window_indices_mx_cached(parents_tuple: tuple[int, ...], keep: int):
    import mlx.core as mx

    return mx.array(_conv_window_indices_cached(parents_tuple, keep), dtype=mx.int32)


@lru_cache(maxsize=512)
def _parents_mx_cached(parents_tuple: tuple[int, ...]):
    import mlx.core as mx

    return mx.array(parents_tuple, dtype=mx.int32)


def _get_tree_gated_delta_kernel():
    """Lazy-create a Metal kernel for one-pass tree GatedDelta recurrence.

    The native sequence kernel handles a linear chain in one launch.  The old
    DDTree path launched a tiny GatedDelta kernel once per tree depth per layer
    (~16 launches × 30 GDN layers on A3B).  This kernel serializes the tree
    nodes inside each (value-head, value-dim) threadgroup and gathers parent
    recurrent state from the already-computed parent node, reducing recurrence
    to one launch per GDN layer.
    """
    global _TREE_GATED_DELTA_KERNEL, _TREE_GATED_DELTA_KERNEL_FAILED
    if _TREE_GATED_DELTA_KERNEL is not None:
        return _TREE_GATED_DELTA_KERNEL
    if _TREE_GATED_DELTA_KERNEL_FAILED:
        return None

    import mlx.core as mx

    if not mx.metal.is_available():
        _TREE_GATED_DELTA_KERNEL_FAILED = True
        return None

    source = r"""
        auto hv_idx = thread_position_in_grid.z;
        auto dv_idx = thread_position_in_grid.y;
        auto dk_lane = thread_position_in_threadgroup.x;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        float state[n_per_t];

        for (int t = 0; t < T; ++t) {
          auto parent_idx = parents[t];

          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            if (parent_idx < 0) {
              auto base_off = (hv_idx * Dv + dv_idx) * Dk + s_idx;
              state[i] = static_cast<float>(base_state[base_off]);
            } else {
              auto parent_off = ((parent_idx * Hv + hv_idx) * Dv + dv_idx) * Dk + s_idx;
              state[i] = static_cast<float>(state_all[parent_off]);
            }
          }

          auto q_ = q + (t * Hk + hk_idx) * Dk;
          auto k_ = k + (t * Hk + hk_idx) * Dk;
          auto v_ = v + (t * Hv + hv_idx) * Dv;
          auto g_ = g + t * Hv;
          auto beta_ = beta + t * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
          }
          out = simd_sum(out);

          if (thread_index_in_simdgroup == 0) {
            y[(t * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }

          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            auto out_off = ((t * Hv + hv_idx) * Dv + dv_idx) * Dk + s_idx;
            state_all[out_off] = static_cast<StT>(state[i]);
          }
        }
    """

    try:
        _TREE_GATED_DELTA_KERNEL = mx.fast.metal_kernel(
            name="ddtree_gated_delta_tree_scalar",
            input_names=["q", "k", "v", "g", "beta", "base_state", "parents", "T"],
            output_names=["y", "state_all"],
            source=source,
        )
    except Exception:
        _TREE_GATED_DELTA_KERNEL_FAILED = True
        return None
    return _TREE_GATED_DELTA_KERNEL


def _get_tree_gated_delta_nostate_kernel():
    """Lazy-create tree GDN kernel that keeps parent states on-chip.

    This variant returns only the per-node recurrent output.  It avoids the
    large ``state_all`` tensor used solely for later cache commit; callers can
    recompute the accepted path state after the tree walk instead.
    """
    global _TREE_GATED_DELTA_NOSTATE_KERNEL, _TREE_GATED_DELTA_NOSTATE_KERNEL_FAILED
    if _TREE_GATED_DELTA_NOSTATE_KERNEL is not None:
        return _TREE_GATED_DELTA_NOSTATE_KERNEL
    if _TREE_GATED_DELTA_NOSTATE_KERNEL_FAILED:
        return None

    import mlx.core as mx

    if not mx.metal.is_available():
        _TREE_GATED_DELTA_NOSTATE_KERNEL_FAILED = True
        return None

    source = r"""
        auto hv_idx = thread_position_in_grid.z;
        auto dv_idx = thread_position_in_grid.y;
        auto dk_lane = thread_position_in_threadgroup.x;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        float state[n_per_t];
        float state_hist[TMAX][n_per_t];

        for (int t = 0; t < T; ++t) {
          auto parent_idx = parents[t];

          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            if (parent_idx < 0) {
              auto base_off = (hv_idx * Dv + dv_idx) * Dk + s_idx;
              state[i] = static_cast<float>(base_state[base_off]);
            } else {
              state[i] = state_hist[parent_idx][i];
            }
          }

          auto q_ = q + (t * Hk + hk_idx) * Dk;
          auto k_ = k + (t * Hk + hk_idx) * Dk;
          auto v_ = v + (t * Hv + hv_idx) * Dv;
          auto g_ = g + t * Hv;
          auto beta_ = beta + t * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
          }
          out = simd_sum(out);

          if (thread_index_in_simdgroup == 0) {
            y[(t * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }

          for (int i = 0; i < n_per_t; ++i) {
            state_hist[t][i] = state[i];
          }
        }
    """

    try:
        _TREE_GATED_DELTA_NOSTATE_KERNEL = mx.fast.metal_kernel(
            name="ddtree_gated_delta_tree_nostate",
            input_names=["q", "k", "v", "g", "beta", "base_state", "parents", "T"],
            output_names=["y"],
            source=source,
        )
    except Exception:
        _TREE_GATED_DELTA_NOSTATE_KERNEL_FAILED = True
        return None
    return _TREE_GATED_DELTA_NOSTATE_KERNEL


def _tree_gated_delta_metal_nostate(
    q,
    k,
    v,
    g,
    beta,
    base_state,
    parents_array,
):
    import mlx.core as mx

    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or g.ndim != 3
        or int(q.shape[0]) != 1
    ):
        return None

    _, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0 or Hv % Hk != 0 or T > 64:
        return None

    kernel = _get_tree_gated_delta_nostate_kernel()
    if kernel is None:
        return None

    input_type = q.dtype
    state_type = base_state.dtype
    try:
        (y,) = kernel(
            inputs=[q, k, v, g, beta, base_state, parents_array, T],
            template=[
                ("InT", input_type),
                ("StT", state_type),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
                ("TMAX", T),
            ],
            grid=(32, Dv, Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(1, T, Hv, Dv)],
            output_dtypes=[input_type],
        )
    except Exception:
        return None
    return y


def _tree_gated_delta_metal(
    q,
    k,
    v,
    g,
    beta,
    base_state,
    parents_array,
):
    import mlx.core as mx

    if (
        mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or g.ndim != 3
        or int(q.shape[0]) != 1
    ):
        return None

    _, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0 or Hv % Hk != 0:
        return None

    kernel = _get_tree_gated_delta_kernel()
    if kernel is None:
        return None

    input_type = q.dtype
    state_type = base_state.dtype
    try:
        y, state_all = kernel(
            inputs=[q, k, v, g, beta, base_state, parents_array, T],
            template=[
                ("InT", input_type),
                ("StT", state_type),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid=(32, Dv, Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(1, T, Hv, Dv), (T, Hv, Dv, Dk)],
            output_dtypes=[input_type, state_type],
        )
    except Exception:
        return None
    return y, state_all


def _tree_state_gdn_forward_fast(
    linear_attn,
    inputs,
    cache,
    *,
    parents: list[int],
    store_state_all: bool = True,
):
    """Fast exact tree-state GDN using vectorized conv + tree Metal kernel."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import compute_g

    B, T, _ = inputs.shape
    if B != 1:
        return None

    keep = int(linear_attn.conv_kernel_size) - 1
    conv_dim = linear_attn.conv_dim

    if cache is not None:
        base_conv_state = cache[0] if cache[0] is not None else mx.zeros(
            (B, keep, conv_dim), dtype=inputs.dtype
        )
        base_state = cache[1] if cache[1] is not None else mx.zeros(
            (B, linear_attn.num_v_heads, linear_attn.head_v_dim,
             linear_attn.head_k_dim),
            dtype=mx.float32,
        )
    else:
        base_conv_state = mx.zeros((B, keep, conv_dim), dtype=inputs.dtype)
        base_state = mx.zeros(
            (B, linear_attn.num_v_heads, linear_attn.head_v_dim,
             linear_attn.head_k_dim),
            dtype=mx.float32,
        )

    # Project all nodes once.
    qkv = linear_attn.in_proj_qkv(inputs)  # (1, T, conv_dim)
    z = linear_attn.in_proj_z(inputs).reshape(
        B, T, linear_attn.num_v_heads, linear_attn.head_v_dim
    )
    b = linear_attn.in_proj_b(inputs)
    a = linear_attn.in_proj_a(inputs)

    # Exact depthwise convolution for every tree node via precomputed ancestor
    # windows.  This removes the old per-depth Python/kernel loop.
    source_table = mx.concatenate([base_conv_state[0], qkv[0]], axis=0)
    parents_tuple = tuple(int(p) for p in parents)
    window_indices = _conv_window_indices_mx_cached(parents_tuple, keep)
    conv_input = mx.take(source_table, window_indices, axis=0)  # (T, K, conv_dim)
    new_conv_states_all = mx.contiguous(conv_input[:, -keep:, :]) if keep > 0 else mx.zeros(
        (T, 0, conv_dim), dtype=inputs.dtype
    )
    conv_weight = linear_attn.conv1d.weight[:, :, 0].T  # (kernel, conv_dim)
    conv_out = nn.silu((conv_input * conv_weight[None, :, :]).sum(axis=1))

    q, k, v = [
        tensor.reshape(1, T, heads, dim)
        for tensor, heads, dim in zip(
            mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
            [linear_attn.num_k_heads, linear_attn.num_k_heads, linear_attn.num_v_heads],
            [linear_attn.head_k_dim, linear_attn.head_k_dim, linear_attn.head_v_dim],
            strict=True,
        )
    ]

    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(linear_attn.A_log, a, linear_attn.dt_bias)

    parents_array = _parents_mx_cached(parents_tuple)
    if store_state_all:
        kernel_result = _tree_gated_delta_metal(q, k, v, g, beta, base_state, parents_array)
        if kernel_result is None:
            return None
        raw_out, state_all = kernel_result
    else:
        raw_out = _tree_gated_delta_metal_nostate(q, k, v, g, beta, base_state, parents_array)
        if raw_out is None:
            return None
        state_all = None

    out = linear_attn.norm(raw_out, z)
    out = linear_attn.out_proj(out.reshape(B, T, -1))

    if state_all is None:
        return out, None, None

    # Return compact all-node tensors.  Commit selects only the final accepted
    # node; building a Python list of T lazy mx.take ops per GDN layer per
    # cycle adds significant graph/Python overhead on A3B.
    return out, {"all": new_conv_states_all}, {"all": state_all}


def _tree_state_gdn_forward(
    linear_attn,
    inputs: "mx.array",
    cache,
    *,
    parents: list[int],
    depth_groups: list[list[int]],
    store_state_all: bool = True,
) -> "tuple[mx.array, list | None, list | None]":
    """Run one GatedDeltaNet layer with per-node state branching.

    Processes nodes depth-by-depth. At each depth, gathers parent GDN
    states, runs the depthwise conv + recurrence update in parallel,
    stores each node's output hidden + recurrent state.

    Keeps per-node GDN states in lists indexed by tree position.
    At commit time, only the final accepted node's state is kept.

    Returns:
        (output_hidden, node_conv_states, node_states)
        - output_hidden: tree-index order hidden output
        - node_conv_states: per-tree-node convolution states after that node
        - node_states: per-tree-node recurrent states after that node
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    B, T, _ = inputs.shape
    if B != 1:
        raise ValueError("tree-state GDN requires batch size 1")

    fast_result = _tree_state_gdn_forward_fast(
        linear_attn,
        inputs,
        cache,
        parents=parents,
        store_state_all=store_state_all,
    )
    if fast_result is not None:
        return fast_result

    # Project inputs
    qkv = linear_attn.in_proj_qkv(inputs)  # (1, T, conv_dim)
    z = linear_attn.in_proj_z(inputs).reshape(
        B, T, linear_attn.num_v_heads, linear_attn.head_v_dim
    )
    b = linear_attn.in_proj_b(inputs)
    a = linear_attn.in_proj_a(inputs)

    keep = int(linear_attn.conv_kernel_size) - 1
    conv_dim = linear_attn.conv_dim

    # Base state from cache (or zeros)
    if cache is not None:
        base_conv_state = cache[0] if cache[0] is not None else mx.zeros(
            (B, keep, conv_dim), dtype=inputs.dtype
        )
        base_state = cache[1] if cache[1] is not None else mx.zeros(
            (B, linear_attn.num_v_heads, linear_attn.head_v_dim,
             linear_attn.head_k_dim),
            dtype=mx.float32,
        )
    else:
        base_conv_state = mx.zeros((B, keep, conv_dim), dtype=inputs.dtype)
        base_state = mx.zeros(
            (B, linear_attn.num_v_heads, linear_attn.head_v_dim,
             linear_attn.head_k_dim),
            dtype=mx.float32,
        )

    conv_weight = linear_attn.conv1d.weight[:, :, 0].T  # (kernel, conv_dim)

    raw_outputs: list = [None] * T
    node_states: list = [None] * T
    node_conv_states: list = [None] * T

    for indices in depth_groups:
        if not indices:
            continue

        # Gather parent states for nodes at this depth
        parent_states = []
        parent_conv_states = []
        for tree_idx in indices:
            pi = int(parents[tree_idx])
            if pi < 0:
                parent_states.append(base_state)
                parent_conv_states.append(base_conv_state)
            else:
                parent_states.append(node_states[pi])
                parent_conv_states.append(node_conv_states[pi])

        state_in = mx.concatenate(parent_states, axis=0)  # (G, ...)
        conv_state = mx.concatenate(parent_conv_states, axis=0)

        # Gather inputs for these nodes
        index_array = mx.array(indices, dtype=mx.int32)
        qkv_step = mx.take(qkv, index_array, axis=1).reshape(
            len(indices), 1, conv_dim
        )

        # Depthwise convolution
        conv_input = mx.concatenate([conv_state, qkv_step], axis=1)
        new_conv_state = (
            mx.contiguous(conv_input[:, -keep:, :])
            if keep > 0
            else mx.zeros((len(indices), 0, conv_dim), dtype=inputs.dtype)
        )
        conv_out = nn.silu(
            (conv_input * conv_weight[None, :, :]).sum(axis=1)[:, None, :]
        )

        # Split Q, K, V from conv output
        q, k, v = [
            tensor.reshape(len(indices), 1, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
                [linear_attn.num_k_heads, linear_attn.num_k_heads, linear_attn.num_v_heads],
                [linear_attn.head_k_dim, linear_attn.head_k_dim, linear_attn.head_v_dim],
                strict=True,
            )
        ]

        # QK norm
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        a_step = mx.take(a, index_array, axis=1).reshape(
            len(indices), 1, linear_attn.num_v_heads
        )
        b_step = mx.take(b, index_array, axis=1).reshape(
            len(indices), 1, linear_attn.num_v_heads
        )

        # Gated delta recurrence
        out, state_out = gated_delta_update(
            q, k, v, a_step, b_step,
            linear_attn.A_log, linear_attn.dt_bias,
            state_in, None,
            use_kernel=not linear_attn.training,
        )

        # Store per-node results
        for group_pos, tree_idx in enumerate(indices):
            raw_outputs[tree_idx] = out[group_pos:group_pos + 1]
            node_states[tree_idx] = state_out[group_pos:group_pos + 1]
            node_conv_states[tree_idx] = new_conv_state[group_pos:group_pos + 1]

    # Concatenate outputs in tree-index order
    out = mx.concatenate(raw_outputs, axis=1)
    out = linear_attn.norm(out, z)
    out = linear_attn.out_proj(out.reshape(B, T, -1))
    return out, node_conv_states, node_states
