import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_entropy_monotone():
    sys.path.insert(0, str(ROOT / "scripts"))
    from hybrid_probe import shannon_entropy
    assert shannon_entropy({"a": 1.0}) == 0.0
    assert abs(shannon_entropy({"a": .5, "b": .5}) - 1.0) < 1e-9  # bits
    assert (shannon_entropy({"a": .5, "b": .5})
            < shannon_entropy({"a": .34, "b": .33, "c": .33}))
    assert shannon_entropy({}) == 0.0
