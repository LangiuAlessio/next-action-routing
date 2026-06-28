"""
Learned neural baseline (GRU) vs the online router + classical pool.

Answers the referee question "is there a learned sequence model, and does the
router/pool beat it?". The GRU (src/baselines/neural_gru.py) is OUT of the
pool, untuned (one fixed config across corpora), trained once per LOUO fold and
frozen at evaluation, scored on the same (ctx, truth) stream and the same
metrics as every other predictor.

D3/D4: full fold loop (pool + 18-cell router grid + GRU), self-contained.
D1:    GRU only (cheap), reusing the cached pool/router per-fold numbers in
       results/d1_folds/*.json and results/route_grid_metrics_D1_full.json
       (re-running the CTW pool on D1 costs ~25 min/fold; the GRU adds nothing
       to that, so we pair the GRU against the already-computed router).

Usage:
    python3 scripts/neural_baseline.py --corpora D4 D3 D1   [--depth 6]
Writes results/neural_baseline.json
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import route_grid_metrics as R                                   # noqa: E402
from data_utils import load_D1, load_D3, load_D4, leave_one_user_out  # noqa: E402
from baselines.neural_gru import NeuralGRU                       # noqa: E402
from stats_paired import boot_ci, wilcoxon_p                     # noqa: E402

GRID_CFGS = [f"{sc}|eta{e}|a{a}" for sc in R.SCOPES for e in R.ETAS for a in R.ALPHAS]


def _paired(router_pf, base_pf, lower_is_better=False):
    sign = -1.0 if lower_is_better else 1.0
    deltas = [sign * (r - b) for r, b in zip(router_pf, base_pf)]
    lo, hi = boot_ci(deltas)
    return {
        "delta": round(R._mean(deltas), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "wilcoxon_p": round(wilcoxon_p(deltas), 4),
        "wins": f"{sum(1 for x in deltas if x > 0)}/{len(deltas)}",
    }


def _neural_fold_metrics(neural, test, depth):
    """Per-fold mean acc1/acc5/mrr/nll for the (already fitted) GRU, scored on
    the same (ctx = seq[max(0,t-depth):t], truth = seq[t+1]) stream."""
    a1 = a5 = mr = nl = 0.0
    n = 0
    for seq in test:
        if len(seq) < 3:
            continue
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - depth):t]
            truth = seq[t + 1]
            v = R._metrics_from_dist(neural.predict(ctx), truth)
            a1 += v[0]; a5 += v[1]; mr += v[2]; nl += v[3]
            n += 1
    if n == 0:
        return None
    return [a1 / n, a5 / n, mr / n, nl / n]


def run_full(loader, depth):
    """D3/D4: pool + 18-cell router grid + GRU, all paired per fold."""
    splits = leave_one_user_out(loader())
    base_names = R.POOL_NAMES + ["ctw_faithful_binary", "neural"]
    base = {nm: {"acc1": [], "acc5": [], "mrr": [], "nll": []} for nm in base_names}
    soft = {c: [] for c in GRID_CFGS}      # per-fold mean acc1 for each cell

    for _uid, train, test in splits:
        pool = R._build_pool(train, depth)
        fb = R.CTWFaithfulBinary(depth=depth); fb.fit(train)
        neural = NeuralGRU(max_depth=depth).fit(train)

        recorded = []
        for seq in test:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - depth):t]
                truth = seq[t + 1]
                dists = [m.predict(ctx) for m in pool]
                recorded.append((dists, fb.predict(ctx), neural.predict(ctx), truth, len(ctx)))
                for m in pool:
                    m.update(ctx, truth)
                fb.update(ctx, truth)
        if not recorded:
            continue
        n = len(recorded)

        # static baselines
        for bi, name in enumerate(R.POOL_NAMES):
            acc = [0.0, 0.0, 0.0, 0.0]
            for dists, _fb, _nu, truth, _h in recorded:
                v = R._metrics_from_dist(dists[bi], truth)
                for j in range(4):
                    acc[j] += v[j]
            for j, key in enumerate(("acc1", "acc5", "mrr", "nll")):
                base[name][key].append(acc[j] / n)
        for slot, key in ((1, "ctw_faithful_binary"), (2, "neural")):
            acc = [0.0, 0.0, 0.0, 0.0]
            for rec in recorded:
                v = R._metrics_from_dist(rec[slot], rec[3])
                for j in range(4):
                    acc[j] += v[j]
            for j, mk in enumerate(("acc1", "acc5", "mrr", "nll")):
                base[key][mk].append(acc[j] / n)

        # router grid (replay over recorded pool dists)
        for sc in R.SCOPES:
            for e in R.ETAS:
                for a in R.ALPHAS:
                    cfg = f"{sc}|eta{e}|a{a}"
                    if sc == "g":
                        routers = {0: R.Router(len(R.POOL_NAMES), eta=e, alpha=a)}
                        pick = lambda h, r=routers: r[0]
                    else:
                        routers = {b: R.Router(len(R.POOL_NAMES), eta=e, alpha=a) for b in range(4)}
                        pick = lambda h, r=routers: r[R._bucket(h)]
                    s1 = 0.0
                    for dists, _fb, _nu, truth, hlen in recorded:
                        rt = pick(hlen)
                        s1 += R._metrics_from_dist(rt.predict_soft(dists), truth)[0]
                        rt.update(dists, truth)
                    soft[cfg].append(s1 / n)

    def agg(d):
        return {k: round(R._mean(v), 4) for k, v in d.items()}

    best_cfg = max(GRID_CFGS, key=lambda c: R._mean(soft[c]))
    router_pf = soft[best_cfg]
    neural_pf = base["neural"]["acc1"]
    best_fixed = max(R.POOL_NAMES, key=lambda nm: R._mean(base[nm]["acc1"]))

    return {
        "folds": len(neural_pf),
        "best_soft_cfg": best_cfg,
        "baselines": {nm: agg(base[nm]) for nm in base_names},
        "router_best_acc1": round(R._mean(router_pf), 4),
        "paired": {
            "router_vs_neural_acc1": _paired(router_pf, neural_pf),
            "neural_vs_ctw_faithful_binary_acc1": _paired(neural_pf, base["ctw_faithful_binary"]["acc1"]),
            "neural_vs_ctw_leaf_acc1": _paired(neural_pf, base["ctw_leaf"]["acc1"]),
            "neural_vs_best_fixed_acc1": {"expert": best_fixed,
                                          **_paired(neural_pf, base[best_fixed]["acc1"])},
        },
        "neural_per_fold_acc1": [round(x, 4) for x in neural_pf],
        "router_per_fold_acc1": [round(x, 4) for x in router_pf],
    }


def run_d1_neural(depth):
    """D1: GRU only, reuse cached pool/router per-fold numbers."""
    splits = list(leave_one_user_out(load_D1()))
    full = json.load(open(ROOT / "results" / "route_grid_metrics_D1_full.json"))
    d1 = full.get("D1", full)
    best_cfg = d1["best_soft_cfg"]

    neural_pf, router_pf, fb_pf, leaf_pf = [], [], [], []
    neural_agg = {"acc1": [], "acc5": [], "mrr": [], "nll": []}
    for fold, (uid, train, test) in enumerate(splits):
        cache = json.load(open(ROOT / "results" / "d1_folds" / f"fold{fold}.json"))
        assert cache["test_uid"] == uid, f"fold {fold} uid mismatch {cache['test_uid']} != {uid}"
        neural = NeuralGRU(max_depth=depth).fit(train)
        m = _neural_fold_metrics(neural, test, depth)
        if m is None:
            continue
        for j, k in enumerate(("acc1", "acc5", "mrr", "nll")):
            neural_agg[k].append(m[j])
        neural_pf.append(m[0])
        router_pf.append(cache["soft"][best_cfg]["acc1"])
        fb_pf.append(cache["base"]["ctw_faithful_binary"]["acc1"])
        leaf_pf.append(cache["base"]["ctw_leaf"]["acc1"])
        print(f"  D1 fold {fold} ({uid}): neural acc1={m[0]:.4f} "
              f"router={router_pf[-1]:.4f} leafKT={leaf_pf[-1]:.4f}", flush=True)

    best_fixed_name = d1["paired"]["acc1_vs_best_fixed"]["expert"]
    return {
        "folds": len(neural_pf),
        "best_soft_cfg": best_cfg,
        "neural_acc1": round(R._mean(neural_agg["acc1"]), 4),
        "neural_acc5": round(R._mean(neural_agg["acc5"]), 4),
        "neural_mrr": round(R._mean(neural_agg["mrr"]), 4),
        "neural_nll": round(R._mean(neural_agg["nll"]), 4),
        "router_best_acc1": round(R._mean(router_pf), 4),
        "ctw_leaf_acc1": round(R._mean(leaf_pf), 4),
        "ctw_faithful_binary_acc1": round(R._mean(fb_pf), 4),
        "paired": {
            "router_vs_neural_acc1": _paired(router_pf, neural_pf),
            "neural_vs_ctw_leaf_acc1": _paired(neural_pf, leaf_pf),
            "neural_vs_ctw_faithful_binary_acc1": _paired(neural_pf, fb_pf),
        },
        "note_best_fixed": best_fixed_name,
        "neural_per_fold_acc1": [round(x, 4) for x in neural_pf],
        "router_per_fold_acc1": [round(x, 4) for x in router_pf],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["D4", "D3", "D1"])
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()
    out = {}
    loaders = {"D3": load_D3, "D4": load_D4}
    for name in args.corpora:
        print(f"=== {name} ===", flush=True)
        if name == "D1":
            out[name] = run_d1_neural(args.depth)
        else:
            out[name] = run_full(loaders[name], args.depth)
        r = out[name]
        print(f"  neural acc1={r.get('neural_acc1', r.get('baselines',{}).get('neural',{}).get('acc1'))} "
              f"| router best acc1={r['router_best_acc1']} "
              f"| router_vs_neural {r['paired']['router_vs_neural_acc1']}", flush=True)

    outp = ROOT / "results" / "neural_baseline.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
