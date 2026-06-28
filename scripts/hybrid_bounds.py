"""Phase-1 G1 probe: entropy-gate theta sweep with oracle and realistic
LLM bounds, using only the PST recompute (no LLM calls).

Gate fires when entropy >= theta. On gated steps:
  - oracle bound: LLM always correct (upper bound on any hybrid);
  - realistic bound: LLM correct with prob = measured zero-shot ReAct
    acc@1 for that dataset (expectation form).
Non-gated steps use PST correctness.

Writes results/hybrid_bounds_<ds>.csv and prints, per dataset, the best
oracle/realistic accuracy at LLM-call-rate <= 0.20 vs the PST baseline.
"""
import json
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
ROWS = json.load(open(ROOT / "results" / "hybrid_probe_rows.json"))
# Measured zero-shot ReAct acc@1 (from results/comparison_<ds>.json).
P_LLM = {"D1": 0.089, "D3": 0.087}


def main():
    summary = {}
    for ds, rs in ROWS.items():
        base = sum(c for _, c in rs) / len(rs)
        ent = sorted(e for e, _ in rs)
        qs = [.50, .60, .70, .75, .80, .85, .90, .95, .98, 1.0]
        grid = sorted(set(ent[min(int(q * len(ent)), len(ent) - 1)] for q in qs))
        best = None
        with open(ROOT / "results" / f"hybrid_bounds_{ds}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["theta", "call_rate", "oracle_acc", "realistic_acc", "pst_base"])
            for th in grid:
                gated = [(e >= th) for e, _ in rs]
                cr = sum(gated) / len(rs)
                orac = sum((1 if g else c) for g, (_, c) in zip(gated, rs)) / len(rs)
                real = sum((P_LLM[ds] if g else c) for g, (_, c) in zip(gated, rs)) / len(rs)
                w.writerow([f"{th:.4f}", f"{cr:.4f}", f"{orac:.4f}",
                            f"{real:.4f}", f"{base:.4f}"])
                if cr <= 0.20 and (best is None or orac > best[1]):
                    best = (cr, orac, real)
        if best is None:  # smallest call-rate row as fallback report
            best = (1.0, base, base)
        summary[ds] = (base, best)
        print(f"{ds}: PST base={base:.4f} | @call_rate<=0.20 "
              f"best oracle={best[1]:.4f} ({best[1]-base:+.4f}) "
              f"realistic={best[2]:.4f} ({best[2]-base:+.4f}) "
              f"call_rate={best[0]:.3f}")
    return summary


if __name__ == "__main__":
    main()
