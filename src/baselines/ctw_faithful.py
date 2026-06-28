"""
Faithful Context-Tree Weighting (Willems, Shtarkov & Tjalkens, 1995).

Unlike ``ctw.CTW`` (a leaf-only Krichevsky-Trofimov read-out over one-vs-rest
binary trees), this is the genuine multi-alphabet weighted mixture:

  * one tree over the full alphabet (M-ary KT estimator, Dirichlet-1/2 at every
    node), not |Σ| independent binary trees;
  * prediction uses the *weighted* coding probability, not the leaf KT:
        P(x_{t+1}=a | h) = Pw_root(h·a) / Pw_root(h),
    where the internal-node recursion is the true CTW mixture
        Pw(s) = 1/2 · Pe(s) + 1/2 · ∏_{children c} Pw(c)
    (product over children, not the average ``ctw.CTW`` uses), and leaves at
    depth D use Pw = Pe.

The KT block factor for symbol a at a node with counts {c_x}, total n, over an
alphabet of size M is (c_a + 1/2)/(n + M/2). Updates and one-symbol-ahead
prediction are O(D·M) per step; everything is kept in the log domain.

The implementation is validated against an independent exact-rational naive
reference in ``tests/test_ctw_faithful.py``.
"""

from __future__ import annotations

import math
from typing import Iterable

_LOG_HALF = math.log(0.5)


def _logaddexp(la: float, lb: float) -> float:
    if la == -math.inf:
        return lb
    if lb == -math.inf:
        return la
    hi, lo = (la, lb) if la >= lb else (lb, la)
    return hi + math.log1p(math.exp(lo - hi))


class _Node:
    __slots__ = ("depth", "counts", "total", "log_pe", "log_pw", "children")

    def __init__(self, depth: int):
        self.depth = depth
        self.counts: dict[str, int] = {}
        self.total = 0
        self.log_pe = 0.0          # log KT block probability (empty -> 0)
        self.log_pw = 0.0          # log weighted probability (empty -> 0)
        self.children: dict[str, "_Node"] = {}


class CTWFaithful:
    """Faithful multi-alphabet CTW predictor of fixed depth D."""

    def __init__(self, depth: int = 6, alphabet: Iterable[str] | None = None):
        self.depth = depth
        self.root = _Node(depth=0)
        if alphabet is not None:
            self._alphabet: list[str] = list(dict.fromkeys(alphabet))
            self._locked = True
        else:
            self._alphabet = []
            self._locked = False

    # -- alphabet management ------------------------------------------------
    @property
    def M(self) -> int:
        return max(len(self._alphabet), 1)

    def _lock_alphabet(self, symbols: Iterable[str]) -> None:
        seen = dict.fromkeys(self._alphabet)
        for s in symbols:
            seen.setdefault(s, None)
        self._alphabet = list(seen)
        self._locked = True

    # -- core updates -------------------------------------------------------
    def _kt_log_factor(self, node: _Node, sym: str) -> float:
        return math.log((node.counts.get(sym, 0) + 0.5) / (node.total + self.M * 0.5))

    def _recompute_log_pw(self, node: _Node) -> None:
        if node.depth == self.depth:                 # leaf
            node.log_pw = node.log_pe
            return
        csum = 0.0
        for c in node.children.values():
            csum += c.log_pw
        node.log_pw = _logaddexp(_LOG_HALF + node.log_pe, _LOG_HALF + csum)

    def update(self, context: list[str], sym: str) -> None:
        if not self._locked:
            self._alphabet.append(sym) if sym not in self._alphabet else None
        rev = list(reversed(context[-self.depth:]))
        path = [self.root]
        node = self.root
        for d in range(len(rev)):
            child = node.children.get(rev[d])
            if child is None:
                child = _Node(depth=d + 1)
                node.children[rev[d]] = child
            node = child
            path.append(node)
        # bottom-up: leaf first so parents see updated child log_pw
        for node in reversed(path):
            node.log_pe += self._kt_log_factor(node, sym)
            node.counts[sym] = node.counts.get(sym, 0) + 1
            node.total += 1
            self._recompute_log_pw(node)

    def fit(self, sequences: Iterable[list[str]]) -> "CTWFaithful":
        seqs = list(sequences)
        if not self._locked:
            self._lock_alphabet(s for seq in seqs for s in seq)
        for seq in seqs:
            for t in range(len(seq)):
                ctx = seq[max(0, t - self.depth): t]
                self.update(ctx, seq[t])
        return self

    # -- prediction ---------------------------------------------------------
    def _path_states(self, context: list[str]) -> list[dict]:
        """Read-only path root..deepest; virtual fresh nodes where missing."""
        rev = list(reversed(context[-self.depth:]))
        states: list[dict] = []
        node: _Node | None = self.root
        for d in range(len(rev) + 1):
            is_leaf = (d == self.depth)
            if node is not None:
                csum = 0.0
                for c in node.children.values():
                    csum += c.log_pw
                states.append({
                    "counts": node.counts, "total": node.total,
                    "log_pe": node.log_pe, "log_pw": node.log_pw,
                    "children_sum": csum, "is_leaf": is_leaf,
                })
                node = node.children.get(rev[d]) if d < len(rev) else None
            else:
                states.append({
                    "counts": {}, "total": 0, "log_pe": 0.0, "log_pw": 0.0,
                    "children_sum": 0.0, "is_leaf": is_leaf,
                })
            if d == len(rev):
                break
        return states

    def predict(self, context: list[str]) -> dict[str, float]:
        if not self._alphabet:
            return {}
        path = self._path_states(context)
        base = path[0]["log_pw"]
        M = self.M
        dist: dict[str, float] = {}
        for a in self._alphabet:
            child_hyp = None
            on_path_child_logpw = None
            for j in range(len(path) - 1, -1, -1):
                node = path[j]
                ca = node["counts"].get(a, 0)
                hyp_log_pe = node["log_pe"] + math.log((ca + 0.5) / (node["total"] + M * 0.5))
                if node["is_leaf"]:
                    hyp_log_pw = hyp_log_pe
                else:
                    csum = node["children_sum"]
                    if child_hyp is not None:
                        csum = csum - on_path_child_logpw + child_hyp
                    hyp_log_pw = _logaddexp(_LOG_HALF + hyp_log_pe, _LOG_HALF + csum)
                on_path_child_logpw = node["log_pw"]
                child_hyp = hyp_log_pw
            dist[a] = math.exp(child_hyp - base)
        s = sum(dist.values())
        if s <= 0:
            return {a: 1.0 / len(self._alphabet) for a in self._alphabet}
        return {a: v / s for a, v in dist.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        dist = self.predict(context)
        return sorted(dist.items(), key=lambda kv: -kv[1])[:k]

    def interpretability_score(self, context: list[str]) -> float:
        if self.depth == 0:
            return 1.0
        return min(len(context), self.depth) / self.depth


class CTWFaithfulBinary:
    """Faithful CTW used in the same one-vs-rest binary decomposition as the
    leaf-KT ``ctw.CTW`` baseline, but with the genuine weighted-mixture
    read-out instead of the leaf-only KT.

    One faithful binary CTW per action: bit 1 = "this action", bit 0 = "any
    other action"; the context is encoded as in ``ctw.CTW`` (1 where the past
    symbol equals the target action). Per-action P(bit=1) is renormalised over
    actions. This isolates the read-out (leaf-KT vs full mixture) at fixed
    alphabet construction, separating it from the single-M-ary-tree variant
    ``CTWFaithful``.
    """

    def __init__(self, depth: int = 6):
        self.depth = depth
        self._vocab: list[str] = []
        self._trees: dict[str, CTWFaithful] = {}

    def _ensure(self, action: str) -> None:
        if action not in self._trees:
            self._trees[action] = CTWFaithful(depth=self.depth, alphabet=["0", "1"])
            self._vocab.append(action)

    def _encode(self, context: list[str], target: str) -> list[str]:
        return ["1" if a == target else "0" for a in context[-self.depth:]]

    def _raw_prob_one(self, context: list[str], action: str) -> float:
        if action not in self._trees:
            return 0.5
        return self._trees[action].predict(self._encode(context, action)).get("1", 0.0)

    def fit(self, sequences) -> "CTWFaithfulBinary":
        for seq in sequences:
            for s in seq:
                self._ensure(s)
            for t in range(len(seq)):
                ctx = seq[max(0, t - self.depth):t]
                self.update(ctx, seq[t])
        return self

    def update(self, context: list[str], next_action: str) -> None:
        self._ensure(next_action)
        for a in self._vocab:
            self._trees[a].update(self._encode(context, a), "1" if next_action == a else "0")

    def predict(self, context: list[str]) -> dict[str, float]:
        if not self._vocab:
            return {}
        raw = {a: self._raw_prob_one(context, a) for a in self._vocab}
        total = sum(raw.values())
        if total <= 0:
            return {a: 1.0 / len(self._vocab) for a in self._vocab}
        return {a: v / total for a, v in raw.items()}

    def predict_top(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.predict(context).items(), key=lambda kv: -kv[1])[:k]

    def interpretability_score(self, context: list[str]) -> float:
        if self.depth == 0:
            return 1.0
        return min(len(context), self.depth) / self.depth
