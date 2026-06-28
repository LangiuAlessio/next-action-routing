"""Phase-1 zero-LLM-cost feasibility probe for the entropy-gated hybrid.

For D1 and D3, leave-one-user-out, reproduce evaluate_model's PST loop
(depth=6, online update) and record per test step:
  (entropy_of_PST_distribution_bits, pst_top1_correct).

No LLM calls. Output: results/hybrid_probe_rows.json = {ds: [[H, c], ...]}.
"""
import sys
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from data_utils import load_D1, load_D3, leave_one_user_out
from run_benchmark import _build_model

DEPTH = 6


def shannon_entropy(dist: dict) -> float:
    """Shannon entropy in bits of a (possibly unnormalised) distribution."""
    ps = [p for p in dist.values() if p > 0]
    s = sum(ps)
    if not ps or s <= 0:
        return 0.0
    return -sum((p / s) * math.log2(p / s) for p in ps)


def capture(loader):
    rows = []
    for _tu, tr, te in leave_one_user_out(loader()):
        m = _build_model("pst", DEPTH)
        m.fit(tr)
        for seq in te:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - DEPTH):t]
                truth = seq[t + 1]
                dist = m.predict(ctx)
                top = max(dist, key=dist.get) if dist else None
                rows.append([shannon_entropy(dist), int(top == truth)])
                m.update(ctx, truth)
    return rows


def main():
    out = {}
    for name, ld in [("D1", load_D1), ("D3", load_D3)]:
        out[name] = capture(ld)
    (ROOT / "results" / "hybrid_probe_rows.json").write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
