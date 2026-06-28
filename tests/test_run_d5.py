import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_d5", str(Path(__file__).parent.parent / "scripts" / "run_d5.py"))
run_d5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_d5)


def test_run_model_on_predictable_sequence():
    # a perfectly periodic session; after warmup a context model should predict acc@1 high,
    # and Markov-0 (frequency only) should be no better on this structured signal.
    ds = {f"s{i}": (f"2026-06-{i:02d}", ["a", "b", "a", "b", "a", "b", "a", "b"]) for i in range(1, 6)}
    splits = run_d5.temporal_split(ds)
    pst = run_d5.run_model("pst", splits)
    m0 = run_d5.run_model("markov0", splits)
    assert set(pst) == {"model", "n_steps", "acc_at_1", "acc_at_5", "nll_bits", "latency_ms"}
    assert pst["n_steps"] > 0
    assert pst["acc_at_1"] >= m0["acc_at_1"]          # context helps on a structured signal
    assert 0.0 <= pst["acc_at_1"] <= 1.0


def test_online_accumulating_beats_floor_and_scales():
    # the O(N) online protocol (depth-windowed accumulation across sessions in time order) —
    # a distinct, scalable protocol from per-session refit. On a structured signal a context
    # model beats the frequency floor; same step count as the per-position scan.
    ds = {f"s{i}": (f"2026-06-{i:02d}", ["a", "b", "c", "a", "b", "c", "a"]) for i in range(1, 6)}
    pst = run_d5.run_model_online("pst", ds)
    m0 = run_d5.run_model_online("markov0", ds)
    assert set(pst) == {"model", "n_steps", "acc_at_1", "acc_at_5", "nll_bits", "latency_ms"}
    assert pst["n_steps"] == sum(len(s) - 1 for _, (_, s) in ds.items())
    assert pst["acc_at_1"] >= m0["acc_at_1"] and 0.0 <= pst["acc_at_1"] <= 1.0
