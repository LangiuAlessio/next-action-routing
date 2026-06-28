"""
Probabilistic Suffix Tree (PST) — C1 baseline.

Ron, Singer & Tishby (1996): variable-memory Markov model with adaptive
pruning. A node labelled by context s is kept if its conditional
distribution differs from its parent by at least `min_gain` (KL-divergence
threshold) and its count exceeds `min_count`.

Usage:
    pst = PST(max_depth=6, min_count=2, min_gain=0.01)
    for seq in training_sequences:
        pst.fit_sequence(seq)
    dist = pst.predict(context)          # -> dict {action: prob}
    top1 = pst.predict_top(context, k=1) # -> list of (action, prob)
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Iterable, Optional


class _Node:
    __slots__ = ("counts", "children", "total")

    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self.total: int = 0
        self.children: dict[str, "_Node"] = {}

    def prob(self, action: str, alpha: float = 0.5) -> float:
        """Laplace-smoothed probability of `action` at this node."""
        vocab = len(self.counts)
        return (self.counts[action] + alpha) / (self.total + alpha * max(vocab, 1))


class PST:
    """Probabilistic Suffix Tree with adaptive depth and Laplace smoothing."""

    def __init__(
        self,
        max_depth: int = 6,
        min_count: int = 2,
        min_gain: float = 0.01,
        alpha: float = 0.5,
    ):
        self.max_depth = max_depth
        self.min_count = min_count
        self.min_gain = min_gain
        self.alpha = alpha
        self.root = _Node()
        self._vocab: set[str] = set()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_sequence(self, seq: list[str]) -> None:
        """Update counts for all suffixes in `seq`."""
        self._vocab.update(seq)
        n = len(seq)
        for t in range(n):
            node = self.root
            # insert action at position t conditioned on suffix seq[t-d..t-1]
            if t < n - 1:
                next_action = seq[t + 1]
            else:
                continue
            # walk backward to build suffix contexts of increasing length
            node.counts[next_action] += 1
            node.total += 1
            for d in range(1, min(t + 1, self.max_depth) + 1):
                ctx_sym = seq[t - d + 1] if t - d + 1 >= 0 else None
                if ctx_sym is None:
                    break
                if ctx_sym not in node.children:
                    node.children[ctx_sym] = _Node()
                node = node.children[ctx_sym]
                node.counts[next_action] += 1
                node.total += 1

    def fit(self, sequences: Iterable[list[str]]) -> "PST":
        for seq in sequences:
            self.fit_sequence(seq)
        self._prune()
        return self

    def _kl_divergence(self, node: _Node, parent: _Node) -> float:
        """KL(node || parent) over shared vocabulary."""
        kl = 0.0
        for a in self._vocab:
            p = node.prob(a, self.alpha)
            q = parent.prob(a, self.alpha)
            kl += p * math.log(p / q + 1e-12)
        return kl

    def _prune(self, node: Optional[_Node] = None, parent: Optional[_Node] = None):
        """Remove nodes that carry insufficient predictive gain."""
        if node is None:
            node = self.root
        to_remove = []
        for sym, child in node.children.items():
            if child.total < self.min_count:
                to_remove.append(sym)
            elif parent is not None and self._kl_divergence(child, node) < self.min_gain:
                to_remove.append(sym)
            else:
                self._prune(child, node)
        for sym in to_remove:
            del node.children[sym]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _find_node(self, context: list[str]) -> _Node:
        """Walk down the tree matching the longest suffix of `context`."""
        node = self.root
        # context is in chronological order; we navigate from most recent
        for sym in reversed(context[-self.max_depth:]):
            child = node.children.get(sym)
            if child is None:
                break
            node = child
        return node

    def predict(self, context: list[str]) -> dict[str, float]:
        """Return probability distribution over next action given context."""
        node = self._find_node(context)
        vocab = self._vocab if self._vocab else set(node.counts.keys())
        total = sum(node.prob(a, self.alpha) for a in vocab)
        return {a: node.prob(a, self.alpha) / total for a in vocab}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (action, probability) pairs."""
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        """Online update: insert one new (context, next_action) observation."""
        self._vocab.add(next_action)
        node = self.root
        node.counts[next_action] += 1
        node.total += 1
        for sym in reversed(context[-self.max_depth:]):
            if sym not in node.children:
                node.children[sym] = _Node()
            node = node.children[sym]
            node.counts[next_action] += 1
            node.total += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def node_count(self, node: Optional[_Node] = None) -> int:
        if node is None:
            node = self.root
        return 1 + sum(self.node_count(c) for c in node.children.values())

    def interpretability_score(self, context: list[str]) -> float:
        """ι: depth of matched node / max_depth (grounded-context ratio)."""
        depth = 0
        node = self.root
        for sym in reversed(context[-self.max_depth:]):
            child = node.children.get(sym)
            if child is None:
                break
            node = child
            depth += 1
        return depth / self.max_depth
