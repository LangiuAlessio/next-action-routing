"""
Evaluation metrics for the PST-vs-LLM benchmark.

Metrics:
  - acc_at_k(pred_dist, truth, k)  → bool
  - bleu_action(pred_seq, ref_seq, n=5) → float
  - latency_ms(fn, *args) → (result, ms)
  - interpretability_score (delegated to PST.interpretability_score)
  - intent_switch_latency(preds, intent_labels) → float
"""

from __future__ import annotations
import time
import math
from typing import Callable, Any


def acc_at_k(
    top_k: list[tuple[str, float]],
    truth: str,
    k: int,
) -> bool:
    """True if `truth` appears in the top-k predictions."""
    return any(action == truth for action, _ in top_k[:k])


def nll_bits(pred_dist: dict, truth: str, floor: float = 1e-12) -> float:
    """Per-symbol negative log-loss in bits: -log2 P(truth).

    Probabilistic metric (Pilato review #6). A predictor that places more
    mass on the realised symbol scores lower, independently of whether it
    flips the top-1. Unseen / zero-probability truths are floored so the
    score stays finite (the floor is the per-symbol coding cost penalty).
    """
    p = pred_dist.get(truth, 0.0)
    if p < floor:
        p = floor
    return -math.log2(p)


def reciprocal_rank(top_k: list[tuple[str, float]], truth: str) -> float:
    """1/rank of `truth` in the ranked predictions (0 if absent).

    Mean over steps gives MRR — a rank-aware view that, unlike acc@1,
    rewards pushing the truth up the list even when it is not top-1.
    """
    for i, (action, _) in enumerate(top_k):
        if action == truth:
            return 1.0 / (i + 1)
    return 0.0


def _ngrams(seq: list[str], n: int) -> list[tuple]:
    return [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]


def bleu_action(pred_seq: list[str], ref_seq: list[str], max_n: int = 4) -> float:
    """
    Sentence-level BLEU over action sequences (no brevity penalty variant).
    Returns geometric mean of n-gram precisions for n=1..max_n.
    """
    if not pred_seq or not ref_seq:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred_seq, n)
        ref_ng = set(_ngrams(ref_seq, n))
        if not pred_ng:
            return 0.0
        matches = sum(1 for ng in pred_ng if ng in ref_ng)
        precision = matches / len(pred_ng)
        if precision == 0:
            return 0.0
        log_sum += math.log(precision)
    brevity = min(1.0, len(pred_seq) / len(ref_seq))
    return brevity * math.exp(log_sum / max_n)


def latency_ms(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run `fn(*args, **kwargs)` and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return result, elapsed


def intent_switch_latency(
    preds: list[str],
    truths: list[str],
    intent_labels: list[int],
    window: int = 10,
) -> float:
    """
    Measures how many steps after an intent switch a model needs before
    returning to >50% accuracy within a rolling window.

    Returns mean adaptation latency (in steps) across all switches.
    Returns None if no switches in sequence.
    """
    if len(preds) != len(truths) != len(intent_labels):
        raise ValueError("preds, truths and intent_labels must have same length")

    switch_points = [
        i for i in range(1, len(intent_labels))
        if intent_labels[i] != intent_labels[i - 1]
    ]
    if not switch_points:
        return None

    latencies = []
    for sw in switch_points:
        for lag in range(window):
            end = sw + lag + window
            if end > len(preds):
                latencies.append(window)
                break
            slice_preds = preds[sw + lag: end]
            slice_truths = truths[sw + lag: end]
            acc = sum(p == t for p, t in zip(slice_preds, slice_truths)) / len(slice_preds)
            if acc > 0.5:
                latencies.append(lag)
                break
        else:
            latencies.append(window)

    return sum(latencies) / len(latencies)
