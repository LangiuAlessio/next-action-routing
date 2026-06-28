"""
Role B: add the GRU to the router pool and show the router TRACKS it.

The neural-baseline run (neural_baseline.py) showed no single predictor wins
across the alphabet range: the GRU wins on D4, the classical router on D3, tie
on D1. Here we add the GRU as an 8th expert and re-run the router grid, to show
the deployment rule absorbs it: the router routes to the GRU where the GRU is
best (D4) without being dragged down where it is bad (D3).

router-7 = classical pool only (the paper's router).
router-8 = classical pool + GRU.
GRU is trained once per fold, frozen at eval (a static expert the router
weights online), bounded depth-6 context, same metrics/stats as everything else.

D3/D4: full, one process.   D1: per-fold (--fold K) for parallelism, then
--aggregate. Writes results/neural_pool.json (D3/D4) and
results/neural_pool_d1/foldK.json (+ merged into results/neural_pool.json on
--aggregate).
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

CFGS = [f"{sc}|eta{e}|a{a}" for sc in R.SCOPES for e in R.ETAS for a in R.ALPHAS]


def _paired(a_pf, b_pf, lower_is_better=False):
    sign = -1.0 if lower_is_better else 1.0
    d = [sign * (x - y) for x, y in zip(a_pf, b_pf)]
    lo, hi = boot_ci(d)
    return {"delta": round(R._mean(d), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "wilcoxon_p": round(wilcoxon_p(d), 4),
            "wins": f"{sum(1 for x in d if x > 0)}/{len(d)}"}


def _stream_fold(train, test, depth):
    """Return per-fold dict: base metrics + soft7[cfg] + soft8[cfg] (all acc1
    means over the fold), recording pool-7 + GRU per-step distributions."""
    pool = R._build_pool(train, depth)
    fb = R.CTWFaithfulBinary(depth=depth); fb.fit(train)
    gru = NeuralGRU(max_depth=depth).fit(train)

    recorded = []
    for seq in test:
        if len(seq) < 3:
            continue
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - depth):t]; truth = seq[t + 1]
            dists = [m.predict(ctx) for m in pool]
            recorded.append((dists, gru.predict(ctx), fb.predict(ctx), truth, len(ctx)))
            for m in pool:
                m.update(ctx, truth)
            fb.update(ctx, truth)
    n = len(recorded)
    if n == 0:
        return None

    base = {}
    for bi, name in enumerate(R.POOL_NAMES):
        acc = [0.0, 0.0, 0.0, 0.0]
        for dists, _g, _fb, truth, _h in recorded:
            v = R._metrics_from_dist(dists[bi], truth)
            for j in range(4):
                acc[j] += v[j]
        base[name] = [acc[j] / n for j in range(4)]
    for slot, key in ((1, "gru"), (2, "ctw_faithful_binary")):
        acc = [0.0, 0.0, 0.0, 0.0]
        for rec in recorded:
            v = R._metrics_from_dist(rec[slot], rec[3])
            for j in range(4):
                acc[j] += v[j]
        base[key] = [acc[j] / n for j in range(4)]

    soft7, soft8 = {}, {}
    for sc in R.SCOPES:
        for e in R.ETAS:
            for a in R.ALPHAS:
                cfg = f"{sc}|eta{e}|a{a}"
                for K, store in ((7, soft7), (8, soft8)):
                    if sc == "g":
                        routers = {0: R.Router(K, eta=e, alpha=a)}
                        pick = lambda h, r=routers: r[0]
                    else:
                        routers = {b: R.Router(K, eta=e, alpha=a) for b in range(4)}
                        pick = lambda h, r=routers: r[R._bucket(h)]
                    s1 = 0.0
                    for dists, gdist, _fb, truth, hlen in recorded:
                        ex = dists if K == 7 else dists + [gdist]
                        rt = pick(hlen)
                        s1 += R._metrics_from_dist(rt.predict_soft(ex), truth)[0]
                        rt.update(ex, truth)
                    store[cfg] = s1 / n
    return {"n": n, "base": base, "soft7": soft7, "soft8": soft8}


def _aggregate(folds):
    """folds: list of per-fold dicts (from _stream_fold). Return summary."""
    names = list(folds[0]["base"].keys())
    base_pf = {nm: [f["base"][nm][0] for f in folds] for nm in names}   # acc1 per fold
    base_all = {nm: {m: round(R._mean([f["base"][nm][j] for f in folds]), 4)
                     for j, m in enumerate(("acc1", "acc5", "mrr", "nll"))} for nm in names}
    soft7_pf = {c: [f["soft7"][c] for f in folds] for c in CFGS}
    soft8_pf = {c: [f["soft8"][c] for f in folds] for c in CFGS}
    best7 = max(CFGS, key=lambda c: R._mean(soft7_pf[c]))
    best8 = max(CFGS, key=lambda c: R._mean(soft8_pf[c]))
    r7, r8, gru = soft7_pf[best7], soft8_pf[best8], base_pf["gru"]
    best_fixed = max(R.POOL_NAMES, key=lambda nm: R._mean(base_pf[nm]))
    return {
        "folds": len(folds),
        "best_cfg_router7": best7, "best_cfg_router8": best8,
        "router7_acc1": round(R._mean(r7), 4),
        "router8_acc1": round(R._mean(r8), 4),
        "gru_acc1": round(R._mean(gru), 4),
        "baselines": base_all,
        "paired": {
            "router8_vs_router7": _paired(r8, r7),
            "router8_vs_gru": _paired(r8, gru),
            "router8_vs_ctw_faithful_binary": _paired(r8, base_pf["ctw_faithful_binary"]),
            "router8_vs_leafKT": _paired(r8, base_pf["ctw_leaf"]),
            "router8_vs_best_fixed": {"expert": best_fixed, **_paired(r8, base_pf[best_fixed])},
        },
        "router7_per_fold": [round(x, 4) for x in r7],
        "router8_per_fold": [round(x, 4) for x in r8],
        "gru_per_fold": [round(x, 4) for x in gru],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--depth", type=int, default=6)
    a = ap.parse_args()
    loaders = {"D3": load_D3, "D4": load_D4, "D1": load_D1}

    d1dir = ROOT / "results" / "neural_pool_d1"
    outp = ROOT / "results" / "neural_pool.json"

    if a.corpus == "D1" and a.fold is not None:
        d1dir.mkdir(exist_ok=True)
        splits = list(leave_one_user_out(load_D1()))
        uid, train, test = splits[a.fold]
        res = _stream_fold(train, test, a.depth)
        res["fold"] = a.fold; res["test_uid"] = uid
        (d1dir / f"fold{a.fold}.json").write_text(json.dumps(res))
        print(f"D1 fold {a.fold} ({uid}) done: r7 best~, r8, gru "
              f"acc1 fold = {res['base']['gru'][0]:.4f}", flush=True)
        return

    if a.corpus == "D1" and a.aggregate:
        folds = [json.load(open(d1dir / f"fold{k}.json")) for k in range(len(list(load_D1())))]
        summ = _aggregate(folds)
        out = json.load(open(outp)) if outp.exists() else {}
        out["D1"] = summ
        outp.write_text(json.dumps(out, indent=2))
        print("D1:", json.dumps(summ["paired"], indent=2), flush=True)
        return

    # D3 / D4 full
    splits = leave_one_user_out(loaders[a.corpus]())
    folds = []
    for uid, train, test in splits:
        r = _stream_fold(train, test, a.depth)
        if r:
            folds.append(r)
    summ = _aggregate(folds)
    out = json.load(open(outp)) if outp.exists() else {}
    out[a.corpus] = summ
    outp.write_text(json.dumps(out, indent=2))
    print(f"=== {a.corpus} ===", flush=True)
    print(f"  router7={summ['router7_acc1']} router8={summ['router8_acc1']} gru={summ['gru_acc1']}", flush=True)
    print(f"  router8_vs_router7 {summ['paired']['router8_vs_router7']}", flush=True)
    print(f"  router8_vs_gru     {summ['paired']['router8_vs_gru']}", flush=True)


if __name__ == "__main__":
    main()
