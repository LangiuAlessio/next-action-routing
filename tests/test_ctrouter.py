import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def test_markovk_predict_update():
    from markov_k import MarkovK
    m = MarkovK(k=1).fit([["a", "b", "a", "b", "a", "b"]])
    d = m.predict(["a"])
    assert isinstance(d, dict) and abs(sum(d.values()) - 1.0) < 1e-9
    assert d.get("b", 0) > d.get("a", 0)
    m.update(["a"], "a")
    assert m.predict(["a"]).get("a", 0) > 0


def _router(**kw):
    from ctrouter import Router
    return Router(n_experts=3, **kw)


def test_hedge_rewards_winner_and_normalises():
    r = _router(eta=1.0, alpha=0.0)
    r.update([{"x": 0.9}, {"x": 0.01}, {"x": 0.01}], "x")
    w = r.weights()
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1] and w[0] > w[2]


def test_fixed_share_floor():
    r = _router(eta=5.0, alpha=0.05)
    for _ in range(50):
        r.update([{"x": 0.99}, {"x": 1e-6}, {"x": 1e-6}], "x")
    assert min(r.weights()) > 0.0


def test_soft_mix_valid_distribution():
    r = _router(eta=1.0, alpha=0.0)
    p = r.predict_soft([{"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}, {"a": 0.5, "b": 0.5}])
    assert abs(sum(p.values()) - 1.0) < 1e-9 and set(p) == {"a", "b"}


def test_hard_top1_picks_argmax_weight():
    r = _router(eta=1.0, alpha=0.0)
    r.update([{"x": 0.9}, {"x": 0.01}, {"x": 0.01}], "x")
    dists = [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}]
    ph = r.predict_hard(dists)
    assert max(ph, key=ph.get) == "a"
