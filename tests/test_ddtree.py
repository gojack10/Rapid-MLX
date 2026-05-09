"""
Deterministic tests for DDTree builder and tree walker.
"""

import numpy as np
import pytest
from vllm_mlx.engine.ddtree import (
    DDTree,
    build_ddtree_tree,
    build_ddtree_tree_from_topk,
    compute_dfs_order,
    follow_verified_tree,
)


# ── toy logit helpers ────────────────────────────────────────────────────────

def _make_logits(token_probs: list[list[float]], vocab_size: int = 16) -> np.ndarray:
    """Build (L, V) logits from per-position token probabilities (L lists)."""
    L = len(token_probs)
    logits = np.full((L, vocab_size), -100.0, dtype=np.float32)
    for pos, probs in enumerate(token_probs):
        for token_id, prob in enumerate(probs):
            logits[pos, token_id] = np.log(max(prob, 1e-9))
    return logits


def _tree_token_ids(tree: DDTree, root_token: int, indices: list[int]) -> list[int]:
    return [
        root_token if idx == 0 else int(tree.node_token_ids[idx - 1])
        for idx in indices
    ]


# ── build_ddtree_tree_from_topk ──────────────────────────────────────────────

def test_empty_budget_returns_empty_tree():
    tree = build_ddtree_tree_from_topk(
        np.array([[0, 1, 2]]), np.log(np.array([[0.5, 0.3, 0.2]])), budget=0
    )
    assert tree.node_count == 0
    assert tree.tree_size == 1
    assert tree.parents == [-1]
    assert tree.child_maps == [{}]
    assert tree.visibility.shape == (1, 1)
    assert tree.visibility[0, 0]


def test_empty_input_returns_empty_tree():
    tree = build_ddtree_tree_from_topk(
        np.empty((0, 0), dtype=np.int64),
        np.empty((0, 0), dtype=np.float32),
        budget=8,
    )
    assert tree.node_count == 0
    assert tree.tree_size == 1


def test_single_token_tree():
    """Budget 1 with a single dominant token should produce 1 node."""
    top_ids = np.array([[5, 3, 7]], dtype=np.int64)  # 1 position, 3 tokens
    top_logprobs = np.log(np.array([[0.5, 0.3, 0.2]], dtype=np.float32))

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=1)
    assert tree.node_count == 1
    assert tree.node_token_ids[0] == 5
    assert tree.node_depths[0] == 1
    assert tree.parents[0] == -1
    assert tree.parents[1] == 0
    assert 5 in tree.child_maps[0]
    assert tree.child_maps[0][5] == 1

    # visibility: root can see root; node1 can see root + node1
    assert tree.visibility.shape == (2, 2)
    assert tree.visibility[0, 0]
    assert not tree.visibility[0, 1]
    assert tree.visibility[1, 0]   # sees root (ancestor)
    assert tree.visibility[1, 1]   # sees self


def test_tree_is_prefix_closed():
    """Every node's parent must precede it in the flat list (prefix-closed)."""
    top_ids = np.array(
        [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=np.int64,
    )
    probs = np.full((4, 3), 1.0 / 3.0, dtype=np.float32)
    top_logprobs = np.log(probs)

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=20)
    assert tree.node_count == 20  # full 4×3=12 tokens drawn from 4 positions, limited to budget

    # Every node's parent index < node index (parent appears before child)
    for idx in range(1, len(tree.parents)):
        assert tree.parents[idx] < idx, f"Node {idx} parent {tree.parents[idx]} is not before it"


def test_visibility_ancestor_only():
    """The visibility matrix must only permit ancestor or self attention."""
    top_ids = np.array(
        [[0, 1], [10, 11], [20, 21]], dtype=np.int64
    )
    probs = np.full((3, 2), 0.5, dtype=np.float32)
    top_logprobs = np.log(probs)

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=6)

    # Check that for every (i, j): if visibility[i,j] and i != j,
    # then j is in the ancestor chain of i.
    def _is_ancestor(child_idx: int, ancestor_idx: int) -> bool:
        while child_idx > 0:
            child_idx = tree.parents[child_idx]
            if child_idx == ancestor_idx:
                return True
        return False

    n = tree.tree_size
    for i in range(n):
        for j in range(n):
            if tree.visibility[i, j]:
                if i == j:
                    continue
                assert _is_ancestor(i, j), (
                    f"Visibility[{i},{j}]=True but {j} is not ancestor of {i}"
                )


def test_child_map_consistent():
    """child_maps must map from parent's token to child index."""
    top_ids = np.array(
        [[5, 3], [15, 13], [25, 23], [35, 33]], dtype=np.int64
    )
    probs = np.full((4, 2), 0.5, dtype=np.float32)
    top_logprobs = np.log(probs)

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=15)

    for parent_idx, child_map in enumerate(tree.child_maps):
        for token_id, child_idx in child_map.items():
            parent_token = (
                -1 if parent_idx == 0 else int(tree.node_token_ids[parent_idx - 1])
            )
            # The child's parent must point back
            assert tree.parents[child_idx] == parent_idx, (
                f"child_map[{parent_idx}][{token_id}]={child_idx} "
                f"but parents[{child_idx}]={tree.parents[child_idx]}"
            )


# ── build_ddtree_tree (from logits) ──────────────────────────────────────────

def test_build_from_logits_single_position():
    logits = _make_logits([[0.5, 0.3, 0.2]], vocab_size=4)
    tree = build_ddtree_tree(logits, budget=2)
    assert tree.node_count == 2
    # First node should be the highest-probability token (id 0)
    assert tree.node_token_ids[0] == 0


def test_build_from_logits_multi_position():
    """3 positions, budget 6 — should build a 6-node tree."""
    # Position 0: token 0 dominant
    # Position 1: token 1 dominant
    # Position 2: token 2 dominant
    logits = _make_logits(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        vocab_size=5,
    )
    tree = build_ddtree_tree(logits, budget=6)
    assert tree.node_count == 6
    # First node at depth 1 should be token 0 (highest prob at position 0)
    assert tree.node_depths[0] == 1
    assert tree.node_token_ids[0] == 0


# ── follow_verified_tree ─────────────────────────────────────────────────────

def test_follow_verified_tree_full_accept():
    """All tokens match → accept everything."""
    # Tree: root(0) → child(1 token=5) → child(2 token=15) → child(3 token=25)
    tree = build_ddtree_tree_from_topk(
        np.array([[5], [15], [25]], dtype=np.int64),
        np.log(np.ones((3, 1), dtype=np.float32)),
        budget=3,
    )
    # Target posterior: matches the drafted tree exactly
    posterior = [5, 15, 25, 999]  # root→5, node1→15, node2→25, node3→999
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == [0, 1, 2, 3]
    assert bonus == 999


def test_follow_verified_tree_partial_accept():
    """First mismatch at node 1 → accept only root."""
    tree = build_ddtree_tree_from_topk(
        np.array([[5], [15], [25]], dtype=np.int64),
        np.log(np.ones((3, 1), dtype=np.float32)),
        budget=3,
    )
    posterior = [999, 15, 25, 999]  # root mismatches immediately
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == [0]
    assert bonus == 999


def test_follow_verified_tree_mid_chain_mismatch():
    """Accept root and node1, mismatch at node2."""
    # Position 0: tokens 5 and 3
    # Position 1: tokens 15 and 13
    # Position 2: tokens 25 and 23
    top_ids = np.array([[5, 3], [15, 13], [25, 23]], dtype=np.int64)
    probs = np.array([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]], dtype=np.float32)
    top_logprobs = np.log(probs)

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=6)
    # Simulate: root→5 (hit), node(5)→15 (hit), node(15)→99 (MISS)
    # Need to figure out indices. root=0, first child of root is token 5 → node1
    # node1's first child is token 15 → node2
    posterior = [5, 15, 99, 0, 0, 0, 0]
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert len(accepted) == 3  # root(0), node(5), node(15)
    assert bonus == 99


def test_follow_verified_tree_empty_tree():
    """No drafted nodes → only root."""
    tree = build_ddtree_tree_from_topk(
        np.empty((0, 0), dtype=np.int64),
        np.empty((0, 0), dtype=np.float32),
        budget=0,
    )
    posterior = [42]
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == [0]
    assert bonus == 42


# ── compute_dfs_order ───────────────────────────────────────────────────────

def test_dfs_order_empty_tree():
    tree = build_ddtree_tree_from_topk(
        np.empty((0, 0), dtype=np.int64),
        np.empty((0, 0), dtype=np.float32),
        budget=0,
    )
    dfs, inv_dfs = compute_dfs_order(tree)
    assert dfs == [0]
    assert inv_dfs == [0]


def test_dfs_order_single_branch():
    """A single vertical chain → DFS should go root → child1 → child2 → ..."""
    tree = build_ddtree_tree_from_topk(
        np.array([[5], [15], [25]], dtype=np.int64),
        np.log(np.ones((3, 1), dtype=np.float32)),
        budget=3,
    )
    dfs, inv_dfs = compute_dfs_order(tree)
    assert dfs == [0, 1, 2, 3]
    assert inv_dfs == [0, 1, 2, 3]


def test_dfs_order_inverse():
    """inv_dfs[dfs[i]] == i for all i."""
    top_ids = np.array([[0, 1], [10, 11], [20, 21]], dtype=np.int64)
    probs = np.full((3, 2), 0.5, dtype=np.float32)
    top_logprobs = np.log(probs)
    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=6)

    dfs, inv_dfs = compute_dfs_order(tree)
    n = tree.tree_size
    assert len(dfs) == n
    assert len(inv_dfs) == n
    for i in range(n):
        assert inv_dfs[dfs[i]] == i
        assert dfs[inv_dfs[i]] == i


# ── regression: high-probability tokens dominate ─────────────────────────────

def test_high_prob_tokens_placed_first():
    """In a budget-limited tree, the highest-probability tokens should be
    the earliest-drafted nodes at each depth."""
    # Position 0: token 5 (0.8), token 3 (0.15), token 7 (0.05)
    # Position 1: token 15 (0.8), token 13 (0.15), token 17 (0.05)
    top_ids = np.array([[5, 3, 7], [15, 13, 17]], dtype=np.int64)
    probs = np.array([[0.8, 0.15, 0.05], [0.8, 0.15, 0.05]], dtype=np.float32)
    top_logprobs = np.log(probs)

    tree = build_ddtree_tree_from_topk(top_ids, top_logprobs, budget=4)

    assert tree.node_token_ids[0] == 5   # highest prob at depth 1
    assert tree.node_token_ids[1] == 15  # child of 5: highest prob at depth 2
    assert tree.node_token_ids[2] == 3   # sibling of 5: next best
