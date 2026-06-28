"""Tests for probabilistic metrics (Pilato review #4/#6): NLL and reciprocal rank."""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_nll_bits_known_values():
    from eval.metrics import nll_bits
    dist = {"a": 0.5, "b": 0.25, "c": 0.25}
    assert abs(nll_bits(dist, "a") - 1.0) < 1e-12
    assert abs(nll_bits(dist, "b") - 2.0) < 1e-12


def test_nll_bits_missing_symbol_uses_floor():
    from eval.metrics import nll_bits
    dist = {"a": 1.0}
    # truth not in dist -> floored, large but finite penalty
    val = nll_bits(dist, "z", floor=1e-12)
    assert math.isfinite(val)
    assert abs(val - (-math.log2(1e-12))) < 1e-9


def test_reciprocal_rank():
    from eval.metrics import reciprocal_rank
    top = [("a", 0.5), ("b", 0.3), ("c", 0.2)]
    assert abs(reciprocal_rank(top, "a") - 1.0) < 1e-12
    assert abs(reciprocal_rank(top, "b") - 0.5) < 1e-12
    assert abs(reciprocal_rank(top, "c") - 1.0 / 3.0) < 1e-12


def test_reciprocal_rank_missing_is_zero():
    from eval.metrics import reciprocal_rank
    top = [("a", 0.5), ("b", 0.5)]
    assert reciprocal_rank(top, "z") == 0.0
