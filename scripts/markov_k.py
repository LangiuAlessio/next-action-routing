"""Fixed-order Markov expert, bench predictor interface."""
from collections import defaultdict


class MarkovK:
    def __init__(self, k: int = 1, alpha: float = 0.5):
        self.k = k
        self.alpha = alpha
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def _key(self, context):
        return tuple(context[-self.k:]) if self.k > 0 else ()

    def fit(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.vocab.add(seq[i])
                self.vocab.add(seq[i + 1])
                self.counts[self._key(seq[:i + 1])][seq[i + 1]] += 1
        return self

    def predict(self, context):
        row = self.counts.get(self._key(context), {})
        V = max(len(self.vocab), 1)
        tot = sum(row.values()) + self.alpha * V
        if tot <= 0:
            return {a: 1.0 / V for a in self.vocab} if self.vocab else {}
        return {a: (row.get(a, 0) + self.alpha) / tot for a in self.vocab}

    def update(self, context, next_action):
        self.vocab.add(next_action)
        self.counts[self._key(context)][next_action] += 1
