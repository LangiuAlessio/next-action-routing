"""Order-0 baseline: predict purely by global action frequency (most-frequent),
ignoring context. The trivial floor every variable-order predictor must beat."""


class Markov0:
    def __init__(self):
        self.counts = {}
        self.total = 0

    def fit(self, sequences):
        for seq in sequences:
            for a in seq:
                self.counts[a] = self.counts.get(a, 0) + 1
                self.total += 1
        return self

    def predict(self, context):
        if self.total == 0:
            return {}
        return {a: c / self.total for a, c in self.counts.items()}

    def predict_top(self, context, k=5):
        t = self.total or 1
        items = sorted(self.counts.items(), key=lambda x: -x[1])[:k]
        return [(a, c / t) for a, c in items]

    def update(self, context, next_action):
        self.counts[next_action] = self.counts.get(next_action, 0) + 1
        self.total += 1
