import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_d5_anomaly", str(Path(__file__).parent.parent / "scripts" / "run_d5_anomaly.py"))
anom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anom)


def test_find_anomalies_flags_the_unexpected_action():
    # a regular "a,b,a,b" behaviour, then a session with a never-seen action "z" mid-stream.
    ds = {f"s{i}": (f"2026-06-{i:02d}", ["a", "b", "a", "b", "a", "b"]) for i in range(1, 8)}
    ds["s9"] = ("2026-06-09", ["a", "b", "z", "b", "a"])   # "z" is the surprise
    out = anom.find_anomalies(ds, threshold_bits=8.0, warmup=12)  # skip cold-start warmup
    assert out and out[0]["action"] == "z"                  # the unseen action is the top anomaly
    assert all(a["nll_bits"] >= 8.0 for a in out)           # threshold respected
    assert out[0]["session"] == "s9"
