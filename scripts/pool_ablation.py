"""Pool ablation for the online router (Pilato review, 2026-06).

Question: which experts actually carry the router's accuracy?  We drop one
expert family at a time (and reduce to Markov-only) and re-measure acc@1.

The recording loop is identical to ``route_grid_metrics.run_corpus`` (experts
update online every step, same skip rules, same metric), so the ``full``
variant reproduces the paper's headline router exactly.  We then replay the
Hedge/Fixed-Share router over the same (eta, alpha) grid (global scope) but
restricted to the kept pool subset, and report the best-cell mean acc@1 over
the leave-one-user-out folds.  Pool order is fixed:

    0 pst   1 ppm   2 ctw_leaf   3 lz   4 m1   5 m2   6 m3

Writes results/pool_ablation.json.  Pure-Python (stdlib + repo modules).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from route_grid_metrics import (                    # noqa: E402
    _build_pool, _metrics_from_dist, _mean,
    POOL_NAMES, ETAS, ALPHAS,
)
from ctrouter import Router                          # noqa: E402
from data_utils import load_D3, load_D4, leave_one_user_out  # noqa: E402

VARIANTS = {
    "full":        [0, 1, 2, 3, 4, 5, 6],
    "no_ctw":      [0, 1, 3, 4, 5, 6],
    "no_pst":      [1, 2, 3, 4, 5, 6],
    "no_ppm":      [0, 2, 3, 4, 5, 6],
    "only_markov": [4, 5, 6],
}


def run_corpus(users, depth):
    splits = leave_one_user_out(users)

    # Record per-fold per-step 7 dists + truth, with experts updating online
    # exactly as in route_grid_metrics.run_corpus.
    folds = []
    for _uid, train, test in splits:
        pool = _build_pool(train, depth)
        recorded = []
        for seq in test:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - depth):t]
                truth = seq[t + 1]
                dists = [m.predict(ctx) for m in pool]
                recorded.append((dists, truth))
                for m in pool:
                    m.update(ctx, truth)
        if recorded:
            folds.append(recorded)

    out = {}
    for vname, idx in VARIANTS.items():
        best_acc1 = -1.0
        best_cell = None
        for e in ETAS:
            for a in ALPHAS:
                fold_acc1 = []
                for recorded in folds:
                    r = Router(len(idx), eta=e, alpha=a)
                    s1 = 0.0
                    n = len(recorded)
                    for dists, truth in recorded:
                        sub = [dists[j] for j in idx]
                        soft_dist = r.predict_soft(sub)
                        m1_, _, _, _ = _metrics_from_dist(soft_dist, truth)
                        s1 += m1_
                        r.update(sub, truth)
                    fold_acc1.append(s1 / n)
                m = _mean(fold_acc1)
                if m > best_acc1:
                    best_acc1 = m
                    best_cell = f"g|eta{e}|a{a}"
        out[vname] = {"acc1": round(best_acc1, 4), "best_cell": best_cell,
                      "n_experts": len(idx)}
    full = out["full"]["acc1"]
    for v in out:
        out[v]["delta_vs_full"] = round(out[v]["acc1"] - full, 4)
    return out


def main():
    depth = 6
    res = {}
    for name, loader in (("D3", load_D3), ("D4", load_D4)):
        print(f"=== {name} ===", flush=True)
        users = loader()
        res[name] = run_corpus(users, depth)
        for v in VARIANTS:
            r = res[name][v]
            print(f"  {v:12s} acc1={r['acc1']:.4f}  "
                  f"delta={r['delta_vs_full']:+.4f}  ({r['best_cell']})", flush=True)
    out_path = ROOT / "results" / "pool_ablation.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
