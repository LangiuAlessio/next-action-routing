"""
Intent clustering for D1 (shell commands) — E09.

Builds co-occurrence matrix over action pairs within a window,
then runs k-means in co-occurrence space to assign each action
to an intent cluster.

For D2 (mobile) we use the ground-truth ACTION_TO_INTENT mapping
from data_utils.py directly.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional


def cooccurrence_matrix(
    sequences: list[list[str]],
    vocab: list[str],
    window: int = 3,
) -> dict[str, dict[str, float]]:
    """
    Build PPMI (Positive Pointwise Mutual Information) co-occurrence matrix.
    Returns {action: {other_action: ppmi_score}}.
    """
    vocab_set = set(vocab)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    marginal: dict[str, int] = defaultdict(int)
    total = 0

    for seq in sequences:
        for t, a in enumerate(seq):
            if a not in vocab_set:
                continue
            for d in range(1, window + 1):
                for b_idx in [t - d, t + d]:
                    if 0 <= b_idx < len(seq):
                        b = seq[b_idx]
                        if b in vocab_set:
                            counts[a][b] += 1
                            marginal[a] += 1
                            total += 1

    ppmi: dict[str, dict[str, float]] = {}
    for a in vocab:
        ppmi[a] = {}
        for b in vocab:
            cnt = counts[a].get(b, 0)
            if cnt == 0:
                ppmi[a][b] = 0.0
            else:
                pmi = math.log((cnt * total) /
                               (marginal[a] * marginal.get(b, 1) + 1e-9))
                ppmi[a][b] = max(0.0, pmi)
    return ppmi


def _vec(action: str, ppmi: dict[str, dict[str, float]], vocab: list[str]) -> list[float]:
    row = ppmi.get(action, {})
    return [row.get(b, 0.0) for b in vocab]


def _dist(v1: list[float], v2: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def kmeans(
    sequences: list[list[str]],
    k: int = 4,
    window: int = 3,
    n_iter: int = 30,
    seed: int = 42,
) -> dict[str, int]:
    """
    Cluster actions into k intent groups via k-means on PPMI vectors.
    Returns {action: cluster_id}.
    """
    import random
    rng = random.Random(seed)

    vocab = sorted({a for seq in sequences for a in seq})
    if not vocab:
        return {}

    ppmi = cooccurrence_matrix(sequences, vocab, window)
    vecs = {a: _vec(a, ppmi, vocab) for a in vocab}

    # initialise centroids with k-means++
    centroids: list[list[float]] = [vecs[rng.choice(vocab)]]
    for _ in range(k - 1):
        dists = [min(_dist(vecs[a], c) for c in centroids) for a in vocab]
        total = sum(dists)
        if total == 0:
            centroids.append(vecs[rng.choice(vocab)])
        else:
            probs = [d / total for d in dists]
            r = rng.random()
            cum = 0.0
            for a, p in zip(vocab, probs):
                cum += p
                if cum >= r:
                    centroids.append(vecs[a])
                    break

    assignment: dict[str, int] = {}
    for _ in range(n_iter):
        # assign
        assignment = {a: min(range(k), key=lambda i: _dist(vecs[a], centroids[i]))
                      for a in vocab}
        # update centroids
        new_centroids = []
        for i in range(k):
            members = [vecs[a] for a, cl in assignment.items() if cl == i]
            if members:
                dim = len(members[0])
                centroid = [sum(m[d] for m in members) / len(members) for d in range(dim)]
            else:
                centroid = centroids[i]
            new_centroids.append(centroid)
        centroids = new_centroids

    return assignment
