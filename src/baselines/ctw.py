"""
Context-Tree Weighting (CTW) — C3 baseline.

Willems, Shtarkov & Tjalkens (1995): Bayesian mixture over all suffix trees
of depth ≤ D, weighted by Krichevsky-Trofimov (KT) estimator at each node.
Update is exact and O(D) per symbol; achieves Rissanen bound asymptotically.

Simplified to binary alphabet internally via one-vs-rest decomposition:
for each action a, P(a|h) is estimated by a separate CTW tree (bit=1 means
"action is a", bit=0 means "action is not a"). Final distribution is
renormalised over all actions.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Iterable


def _kt_update(n0: int, n1: int) -> float:
    """KT estimator: P(x_{n+1}=1 | n0 zeros, n1 ones) = (n1 + 0.5) / (n0 + n1 + 1)."""
    return (n1 + 0.5) / (n0 + n1 + 1.0)


class _CTWNode:
    __slots__ = ("n0", "n1", "children", "log_pe", "log_pw")

    def __init__(self):
        self.n0: int = 0          # count of 0-bits seen at this node
        self.n1: int = 1          # count of 1-bits seen
        self.children: dict[int, "_CTWNode"] = {}
        self.log_pe: float = 0.0  # log KT (leaf) probability
        self.log_pw: float = 0.0  # log weighted probability


class CTWBinary:
    """CTW predictor for a binary alphabet {0, 1} given depth D."""

    def __init__(self, depth: int = 6):
        self.depth = depth
        self.root = _CTWNode()

    def _get_or_create(self, ctx_bits: list[int]) -> list[_CTWNode]:
        """Return path of nodes from root following ctx_bits (reversed = oldest first)."""
        nodes = [self.root]
        node = self.root
        for bit in reversed(ctx_bits[-self.depth:]):
            if bit not in node.children:
                node.children[bit] = _CTWNode()
            node = node.children[bit]
            nodes.append(node)
        return nodes

    def update(self, ctx_bits: list[int], x: int) -> None:
        """Update tree with observation x ∈ {0,1} given context ctx_bits."""
        nodes = self._get_or_create(ctx_bits)
        # update from leaf to root
        for node in reversed(nodes):
            p_kt = _kt_update(node.n0, node.n1) if x == 1 else (1 - _kt_update(node.n0, node.n1))
            node.log_pe += math.log(p_kt + 1e-300)
            if not node.children:
                node.log_pw = node.log_pe
            else:
                # weighted mixture: 0.5 * Pe + 0.5 * Pw_children
                child_log_pw = sum(
                    c.log_pw for c in node.children.values()
                ) / max(len(node.children), 1)
                log_half = math.log(0.5)
                log_mix = _log_add(log_half + node.log_pe, log_half + child_log_pw)
                node.log_pw = log_mix
            if x == 1:
                node.n1 += 1
            else:
                node.n0 += 1

    def predict_prob1(self, ctx_bits: list[int]) -> float:
        """P(next=1 | context) via KT at root after following context."""
        nodes = self._get_or_create(ctx_bits)
        leaf = nodes[-1]
        return _kt_update(leaf.n0, leaf.n1)


def _log_add(la: float, lb: float) -> float:
    if la > lb:
        return la + math.log1p(math.exp(lb - la) + 1e-300)
    return lb + math.log1p(math.exp(la - lb) + 1e-300)


class CTW:
    """
    CTW predictor over a finite action alphabet.

    Uses one binary CTW per action (one-vs-rest), then renormalises.
    """

    def __init__(self, depth: int = 6):
        self.depth = depth
        self._vocab: set[str] = set()
        self._trees: dict[str, CTWBinary] = {}

    def _ensure_action(self, action: str) -> None:
        if action not in self._trees:
            self._trees[action] = CTWBinary(self.depth)
            self._vocab.add(action)

    def _encode_ctx(self, context: list[str], target: str) -> list[int]:
        """Binary context: 1 if context action == target, else 0."""
        return [1 if a == target else 0 for a in context[-self.depth:]]

    def fit_sequence(self, seq: list[str]) -> None:
        self._vocab.update(seq)
        for a in seq:
            self._ensure_action(a)
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - self.depth + 1): t + 1]
            next_a = seq[t + 1]
            for a in self._vocab:
                bits = self._encode_ctx(ctx, a)
                x = 1 if next_a == a else 0
                self._trees[a].update(bits, x)

    def fit(self, sequences: Iterable[list[str]]) -> "CTW":
        for seq in sequences:
            self.fit_sequence(seq)
        return self

    def predict(self, context: list[str]) -> dict[str, float]:
        if not self._vocab:
            return {}
        raw = {}
        for a in self._vocab:
            bits = self._encode_ctx(context, a)
            raw[a] = self._trees[a].predict_prob1(bits)
        total = sum(raw.values())
        if total == 0:
            return {a: 1.0 / len(self._vocab) for a in self._vocab}
        return {a: v / total for a, v in raw.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        self._ensure_action(next_action)
        for a in self._vocab:
            bits = self._encode_ctx(context, a)
            x = 1 if next_action == a else 0
            self._trees[a].update(bits, x)

    def interpretability_score(self, context: list[str]) -> float:
        return min(len(context), self.depth) / self.depth
