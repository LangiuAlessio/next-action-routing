"""
LZ78-based universal predictor — C4 baseline.

Feder, Merhav & Gutman (1992): incremental LZ78 parsing achieves
finite-state predictability within a constant additive factor of the
optimal predictor, without parametric assumptions on the source.

Implementation: LZ78 trie where each leaf is a new phrase.
Prediction at step t = distribution over next symbols at the deepest
LZ78 trie node matching the current parsing context.
Online: trie grows monotonically as new phrases are discovered.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable


class _LZNode:
    __slots__ = ("children", "counts", "total", "phrase_id")

    def __init__(self, phrase_id: int = -1):
        self.children: dict[str, "_LZNode"] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.total: int = 0
        self.phrase_id: int = phrase_id

    def prob(self, action: str, alpha: float = 0.5, vocab_size: int = 1) -> float:
        return (self.counts[action] + alpha) / (self.total + alpha * vocab_size)


class LZPredictor:
    """
    LZ78 trie predictor.

    During training, parses each sequence into LZ78 phrases and records
    the next symbol after each phrase boundary. At inference, descends
    the trie as far as possible given the current context and returns
    the empirical distribution at the deepest matching node.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.root = _LZNode(phrase_id=0)
        self._n_phrases: int = 1
        self._vocab: set[str] = set()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_sequence(self, seq: list[str]) -> None:
        self._vocab.update(seq)
        node = self.root
        for t, sym in enumerate(seq):
            # record sym as a "next symbol" observation at current node
            # (before we decide whether to extend or branch)
            if t < len(seq) - 1:
                node.counts[seq[t + 1]] += 1
                node.total += 1

            if sym not in node.children:
                # new phrase: create leaf, reset to root
                new_node = _LZNode(phrase_id=self._n_phrases)
                self._n_phrases += 1
                node.children[sym] = new_node
                node = self.root
            else:
                node = node.children[sym]

    def fit(self, sequences: Iterable[list[str]]) -> "LZPredictor":
        for seq in sequences:
            self.fit_sequence(seq)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _find_node(self, context: list[str]) -> _LZNode:
        """Descend trie matching the longest suffix of context."""
        node = self.root
        for sym in context:
            child = node.children.get(sym)
            if child is None:
                break
            node = child
        return node

    def predict(self, context: list[str]) -> dict[str, float]:
        vocab = self._vocab
        if not vocab:
            return {}
        node = self._find_node(context)
        vs = len(vocab)
        raw = {a: node.prob(a, self.alpha, vs) for a in vocab}
        total = sum(raw.values())
        return {a: v / total for a, v in raw.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    def update(self, context: list[str], next_action: str) -> None:
        """Online update: extend trie with new observation."""
        self._vocab.add(next_action)
        node = self._find_node(context)
        node.counts[next_action] += 1
        node.total += 1
        # extend trie with the last context symbol if possible
        if context:
            last = context[-1]
            if last not in node.children:
                node.children[last] = _LZNode(phrase_id=self._n_phrases)
                self._n_phrases += 1

    def interpretability_score(self, context: list[str]) -> float:
        """Depth reached in trie / len(context) (capped at 1)."""
        depth = 0
        node = self.root
        for sym in context:
            child = node.children.get(sym)
            if child is None:
                break
            node = child
            depth += 1
        return depth / max(len(context), 1)

    def phrase_count(self) -> int:
        return self._n_phrases
