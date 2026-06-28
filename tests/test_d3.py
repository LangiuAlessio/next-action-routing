import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_sessionise_idle_gap():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_d3_eclipse import sessionise
    # (ts_seconds, cmd); a gap > 1800 s splits the session
    evs = [(0, "a"), (10, "b"), (10_000, "c"), (10_010, "d")]
    assert sessionise(evs, idle_gap_s=1800) == [["a", "b"], ["c", "d"]]


def test_normalise_cmd_id():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_d3_eclipse import normalise_cmd_id
    assert normalise_cmd_id("org.eclipse.ui.edit.text.save") == "edit.text.save"
    assert normalise_cmd_id("org.eclipse.jdt.ui.JavaPerspective") == "jdt.ui.javaperspective"
    assert normalise_cmd_id("save") == "save"
    assert normalise_cmd_id("") is None
    assert normalise_cmd_id("   ") is None


def test_load_d3_structure():
    """Requires data/D3-eclipse/USER*.data to exist (Task A1 Step 5)."""
    d3 = ROOT / "data" / "D3-eclipse"
    if not list(d3.glob("USER*.data")):
        import pytest
        pytest.skip("D3 not built yet (run build_d3_eclipse.py)")
    sys.path.insert(0, str(ROOT / "src"))
    from data_utils import load_D3
    users = load_D3()
    assert isinstance(users, dict) and len(users) >= 5
    some = next(iter(users.values()))
    assert isinstance(some, list) and isinstance(some[0], list)
    assert all(isinstance(tok, str) for tok in some[0])


def test_react_cap_and_resume(tmp_path):
    """LLM cost-cap + checkpoint/resume logic, exercised with the fast
    PST model (no LLM calls): cap stops at N folds; rerun resumes from
    the result-JSON checkpoint without recomputing done folds."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    rb = importlib.import_module("run_benchmark")
    from data_utils import load_D3
    users = load_D3()
    assert len(users) >= 4
    ckpt = tmp_path / "ck.json"

    r1 = rb.evaluate_model("pst", users, "D3", max_folds=2, ckpt_path=ckpt)
    assert r1["n_users"] == 2
    assert ckpt.exists()
    import json as _j
    saved = _j.loads(ckpt.read_text())
    assert saved["n_users"] == 2
    done_first = [u["user"] for u in saved["per_user"]]

    # Resume: cap raised to 4 -> keeps the 2 done verbatim, adds 2 more.
    r2 = rb.evaluate_model("pst", users, "D3", max_folds=4, ckpt_path=ckpt)
    assert r2["n_users"] == 4
    assert [u["user"] for u in r2["per_user"][:2]] == done_first
    assert r2["per_user"][0] == saved["per_user"][0]  # resumed, not recomputed
