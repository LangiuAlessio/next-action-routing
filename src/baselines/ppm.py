"""
Prediction by Partial Matching (PPM) — C2 baseline.

Cleary & Witten (1984): compression-prediction duality via escape mechanism.
At order k, if context not seen, escape to order k-1 with weight = (novel / total).
Implements PPM-C variant (Moffat 1990): escape probability = unique_symbols / total.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Optional


class _Ctx:
    __slots__ = ("counts", "total", "unique")

    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self.total: int = 0
        self.unique: int = 0  # distinct symbols seen (for PPM-C escape)

    def escape_prob(self) -> float:
        if self.total == 0:
            return 1.0
        return self.unique / (self.total + self.unique)  # PPM-C

    def prob(self, action: str) -> float:
        if self.counts[action] == 0 or self.total == 0:
            return 0.0
        return self.counts[action] / (self.total + self.unique)

    def update(self, action: str):
        if self.counts[action] == 0:
            self.unique += 1
        self.counts[action] += 1
        self.total += 1


class PPM:
    """PPM-C predictor up to order `max_order`."""

    def __init__(self, max_order: int = 5):
        self.max_order = max_order
        # contexts[(sym1,...,symK)] -> _Ctx
        self.contexts: dict[tuple, _Ctx] = defaultdict(_Ctx)
        self._vocab: set[str] = set()

    # ------------------------------------------------------------------
    def fit_sequence(self, seq: list[str]) -> None:
        self._vocab.update(seq)
        n = len(seq)
        for t in range(n - 1):
            next_action = seq[t + 1]
            # update contexts of length 0..max_order
            for k in range(0, min(t + 1, self.max_order) + 1):
                ctx = tuple(seq[t - k + 1: t + 1]) if k > 0 else ()
                self.contexts[ctx].update(next_action)

    def fit(self, sequences: Iterable[list[str]]) -> "PPM":
        for seq in sequences:
            self.fit_sequence(seq)
        return self

    # ------------------------------------------------------------------
    def predict(self, context: list[str]) -> dict[str, float]:
        """
        Blend predictions from order K down to 0 via escape mechanism.
        Returns distribution over self._vocab.
        """
        vocab = self._vocab
        if not vocab:
            return {}

        # start from highest order available
        dist: dict[str, float] = {}
        excluded: set[str] = set()
        weight = 1.0

        for k in range(min(len(context), self.max_order), -1, -1):
            ctx_key = tuple(context[-k:]) if k > 0 else ()
            node = self.contexts.get(ctx_key)
            if node is None or node.total == 0:
                continue

            esc = node.escape_prob()
            for a in vocab:
                if a not in excluded:
                    p = node.prob(a)
                    if p > 0:
                        dist[a] = dist.get(a, 0.0) + weight * (1 - esc) * p

            new_seen = {a for a in vocab if node.counts.get(a, 0) > 0}
            excluded |= new_seen
            weight *= esc
            if weight < 1e-10:
                break

        # uniform over remaining unseen symbols
        unseen = vocab - excluded
        if unseen and weight > 0:
            for a in unseen:
                dist[a] = dist.get(a, 0.0) + weight / len(unseen)

        total = sum(dist.values())
        if total == 0:
            return {a: 1.0 / len(vocab) for a in vocab}
        return {a: v / total for a, v in dist.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        self._vocab.add(next_action)
        for k in range(0, min(len(context), self.max_order) + 1):
            ctx = tuple(context[-k:]) if k > 0 else ()
            self.contexts[ctx].update(next_action)

    def interpretability_score(self, context: list[str]) -> float:
        """Depth of highest non-empty context / max_order."""
        for k in range(min(len(context), self.max_order), 0, -1):
            ctx = tuple(context[-k:])
            node = self.contexts.get(ctx)
            if node and node.total > 0:
                return k / self.max_order
        return 0.0
