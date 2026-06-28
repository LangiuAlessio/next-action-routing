"""Online per-user router: Hedge multiplicative weights + Fixed-Share
drift mixing. Soft mixture and hard top-1 (MoE-faithful) read-out.
A fresh Router() per test user = per-user reset (gradient-free)."""
import math


class Router:
    def __init__(self, n_experts: int, eta: float = 0.5, alpha: float = 0.0):
        self.n = n_experts
        self.eta = eta
        self.alpha = alpha
        self.w = [1.0 / n_experts] * n_experts

    def weights(self):
        return list(self.w)

    def predict_soft(self, dists):
        keys = set()
        for d in dists:
            keys.update(d)
        p = {a: sum(self.w[i] * dists[i].get(a, 0.0) for i in range(self.n))
             for a in keys}
        s = sum(p.values())
        if s > 0:
            return {a: v / s for a, v in p.items()}
        return {a: 1.0 / len(keys) for a in keys} if keys else {}

    def predict_hard(self, dists):
        return dict(dists[max(range(self.n), key=lambda i: self.w[i])])

    def update(self, dists, truth):
        for i in range(self.n):
            loss = -math.log(dists[i].get(truth, 0.0) + 1e-12)
            self.w[i] *= math.exp(-self.eta * loss)
        s = sum(self.w) or 1.0
        self.w = [x / s for x in self.w]
        if self.alpha > 0.0:                       # Fixed-Share drift mix
            u = 1.0 / self.n
            self.w = [(1 - self.alpha) * x + self.alpha * u for x in self.w]
            s = sum(self.w)
            self.w = [x / s for x in self.w]
