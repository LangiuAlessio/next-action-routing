import importlib.util
from pathlib import Path


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).parent.parent / "scripts" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_d5_router = _load("run_d5_router")
run_d5 = _load("run_d5")


def test_router_runs_and_beats_floor_on_structured_signal():
    ds = {f"s{i}": (f"2026-06-{i:02d}", ["a", "b", "a", "b", "a", "b", "a", "b"]) for i in range(1, 6)}
    splits = run_d5.temporal_split(ds)
    soft = run_d5_router.run_router(splits, readout="soft")
    hard = run_d5_router.run_router(splits, readout="hard")
    m0 = run_d5.run_model("markov0", splits)
    for r in (soft, hard):
        assert set(r) == {"model", "n_steps", "acc_at_1", "acc_at_5", "nll_bits", "latency_ms"}
        assert r["n_steps"] > 0 and 0.0 <= r["acc_at_1"] <= 1.0
    assert soft["acc_at_1"] >= m0["acc_at_1"]   # the mix beats the frequency floor
    assert soft["model"] == "router-soft" and hard["model"] == "router-hard"
