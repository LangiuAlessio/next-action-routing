"""
Tests for the faithful Willems–Shtarkov–Tjalkens (1995) CTW predictor.

Correctness anchor: an independent, deliberately naive recursive CTW that
recomputes the root weighted block probability Pw from scratch for the whole
sequence using exact rational arithmetic (fractions.Fraction). The production
CTWFaithful (incremental, log-domain, cached) must match it.

Convention (shared by reference and production): when predicting x_t the
context is seq[max(0, t-depth):t]. Every emitted symbol updates the KT
estimator at EVERY node along its root->leaf path (depths 0..min(len(ctx),D)).
A node at depth exactly D is a leaf (Pw = Pe); a node at depth < D is internal
(Pw = 1/2 Pe + 1/2 * prod_{children present} Pw_child); absent children
contribute an empty block (Pw = 1) and are skipped in the product.
"""
import sys
import random
from pathlib import Path
from collections import defaultdict
from fractions import Fraction

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Independent naive reference (exact rational CTW).
# ---------------------------------------------------------------------------
def _kt_block(counts: dict, M: int) -> Fraction:
    """Exact KT block probability for a node with the given symbol counts.

    Pe = [prod_x prod_{j=0}^{c_x-1}(j+1/2)] / [prod_{i=0}^{n-1}(i + M/2)].
    """
    num = Fraction(1)
    for c in counts.values():
        for j in range(c):
            num *= Fraction(2 * j + 1, 2)
    den = Fraction(1)
    total = sum(counts.values())
    for i in range(total):
        den *= Fraction(2 * i + M, 2)
    return num / den


def _build_node_counts(seq, depth):
    """node (most-recent-first context tuple) -> {symbol: count}."""
    node_counts = defaultdict(lambda: defaultdict(int))
    for t in range(len(seq)):
        ctx = seq[max(0, t - depth):t]
        rev = list(reversed(ctx))            # most recent first
        sym = seq[t]
        for d in range(0, min(len(rev), depth) + 1):
            node_counts[tuple(rev[:d])][sym] += 1
    return node_counts


def naive_root_pw(seq, depth, alphabet) -> Fraction:
    """Exact CTW root weighted block probability of the whole sequence."""
    M = len(alphabet)
    node_counts = _build_node_counts(seq, depth)

    def pw(node):
        pe = _kt_block(node_counts.get(node, {}), M)
        if len(node) == depth:
            return pe
        prod = Fraction(1)
        for s in alphabet:
            child = node + (s,)
            if child in node_counts:
                prod *= pw(child)
        return (pe + prod) / 2

    return pw(())


def naive_predict(seq, depth, alphabet) -> dict:
    """Exact CTW predictive distribution P(next=a | seq) for each a in alphabet."""
    base = naive_root_pw(seq, depth, alphabet)
    out = {}
    for a in alphabet:
        out[a] = float(naive_root_pw(seq + [a], depth, alphabet) / base)
    return out


# ---------------------------------------------------------------------------
# Reference sanity (the reference must itself be sensible).
# ---------------------------------------------------------------------------
def test_reference_kt_block_known_values():
    # Binary KT block probs: Pe(1,1)=1/8, Pe(2,1)=1/16.
    assert _kt_block({"0": 1, "1": 1}, 2) == Fraction(1, 8)
    assert _kt_block({"0": 2, "1": 1}, 2) == Fraction(1, 16)
    # Empty block = 1.
    assert _kt_block({}, 2) == Fraction(1)


def test_reference_predictive_sums_to_one():
    seq = list("0110100")
    dist = naive_predict(seq, depth=3, alphabet=["0", "1"])
    assert abs(sum(dist.values()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Production CTWFaithful.
# ---------------------------------------------------------------------------
def _fit_predict(seq, depth, alphabet):
    """Stream seq into a fresh CTWFaithful, return predictive dist after seq."""
    from baselines.ctw_faithful import CTWFaithful
    m = CTWFaithful(depth=depth, alphabet=alphabet)
    for t in range(len(seq)):
        ctx = seq[max(0, t - depth):t]
        m.update(ctx, seq[t])
    return m.predict(seq[max(0, len(seq) - depth):])


def test_predictive_sums_to_one():
    dist = _fit_predict(list("0110100"), depth=3, alphabet=["0", "1"])
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_depth0_reduces_to_plain_kt():
    # Depth 0: only the root leaf -> pure KT over global counts.
    from baselines.ctw_faithful import CTWFaithful
    m = CTWFaithful(depth=0, alphabet=["a", "b"])
    for s in list("aab"):
        m.update([], s)
    dist = m.predict([])
    # counts a=2,b=1,total=3,M=2 -> P(a)=(2.5)/(3+1)=0.625, P(b)=1.5/4=0.375
    assert abs(dist["a"] - 0.625) < 1e-9
    assert abs(dist["b"] - 0.375) < 1e-9


def test_multisymbol_kt_predictive():
    from baselines.ctw_faithful import CTWFaithful
    m = CTWFaithful(depth=0, alphabet=["a", "b", "c"])
    m.update([], "a")
    dist = m.predict([])
    # M=3, counts a=1,total=1 -> P(a)=(1.5)/(1+1.5)=0.6, P(b)=P(c)=0.5/2.5=0.2
    assert abs(dist["a"] - 0.6) < 1e-9
    assert abs(dist["b"] - 0.2) < 1e-9
    assert abs(dist["c"] - 0.2) < 1e-9


def test_matches_naive_binary_depth1():
    random.seed(11)
    alphabet = ["0", "1"]
    for _ in range(40):
        n = random.randint(3, 9)
        seq = [random.choice(alphabet) for _ in range(n)]
        ref = naive_predict(seq, depth=1, alphabet=alphabet)
        got = _fit_predict(seq, depth=1, alphabet=alphabet)
        for a in alphabet:
            assert abs(got.get(a, 0.0) - ref[a]) < 1e-9, (seq, a, got, ref)


def test_matches_naive_binary_depth2():
    random.seed(23)
    alphabet = ["0", "1"]
    for _ in range(40):
        n = random.randint(4, 11)
        seq = [random.choice(alphabet) for _ in range(n)]
        ref = naive_predict(seq, depth=2, alphabet=alphabet)
        got = _fit_predict(seq, depth=2, alphabet=alphabet)
        for a in alphabet:
            assert abs(got.get(a, 0.0) - ref[a]) < 1e-9, (seq, a, got, ref)


def test_matches_naive_multisymbol_depth2():
    random.seed(37)
    alphabet = ["a", "b", "c"]
    for _ in range(40):
        n = random.randint(5, 12)
        seq = [random.choice(alphabet) for _ in range(n)]
        # ensure the model has discovered the full alphabet so M matches
        seq = alphabet + seq
        ref = naive_predict(seq, depth=2, alphabet=alphabet)
        got = _fit_predict(seq, depth=2, alphabet=alphabet)
        for a in alphabet:
            assert abs(got.get(a, 0.0) - ref[a]) < 1e-9, (seq, a, got, ref)


def test_faithful_binary_ovr_sums_to_one_and_differs_from_leafkt():
    """One-vs-rest faithful CTW: proper distribution, and the mixture readout
    differs from the leaf-KT one-vs-rest on a sequence where mixing matters."""
    from baselines.ctw_faithful import CTWFaithfulBinary
    from baselines.ctw import CTW
    seq = list("abacabacabac")
    fb = CTWFaithfulBinary(depth=3)
    leaf = CTW(depth=3)
    for t in range(len(seq)):
        ctx = seq[max(0, t - 3):t]
        fb.update(ctx, seq[t]); leaf.update(ctx, seq[t])
    ctx = seq[-3:]
    d = fb.predict(ctx)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    # mixture readout must differ from leaf-only readout
    assert abs(d.get("a", 0) - leaf.predict(ctx).get("a", 0)) > 1e-6


def test_faithful_binary_per_action_tree_matches_reference():
    """Each per-action binary tree is a faithful binary CTW, so its P(bit=1)
    must match the exact rational reference run on the one-vs-rest bit stream."""
    from baselines.ctw_faithful import CTWFaithfulBinary
    seq = list("ababaabb")
    fb = CTWFaithfulBinary(depth=2)
    for t in range(len(seq)):
        ctx = seq[max(0, t - 2):t]
        fb.update(ctx, seq[t])
    raw1 = fb._raw_prob_one(seq[max(0, len(seq) - 2):], "a")
    bitseq = ["1" if s == "a" else "0" for s in seq]
    ref1 = naive_predict(bitseq, depth=2, alphabet=["0", "1"])["1"]
    assert abs(raw1 - ref1) < 1e-9


def test_differs_from_leaf_kt_readout():
    """Regression guard: the faithful mixture must NOT equal the broken
    leaf-only KT read-out on a sequence where mixing matters."""
    from baselines.ctw_faithful import CTWFaithful
    from baselines.ctw import CTW
    seq = list("abababab")
    faithful = CTWFaithful(depth=3, alphabet=["a", "b"])
    leaf = CTW(depth=3)
    for t in range(len(seq)):
        ctx = seq[max(0, t - 3):t]
        faithful.update(ctx, seq[t])
        leaf.update(ctx, seq[t])
    ctx = seq[-3:]
    df = faithful.predict(ctx)
    dl = leaf.predict(ctx)
    assert abs(df.get("a", 0) - dl.get("a", 0)) > 1e-6
