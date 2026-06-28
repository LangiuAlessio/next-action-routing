"""Aggregate the 9 per-fold D1 JSONs (from run_d1_fold.py) into the full D1 result, replicating the
exact aggregation + paired stats of route_grid_metrics.run_corpus (best cell, grid robustness,
paired vs CTW/leaf/faithful/best-fixed/Markov-1 with Wilcoxon + bootstrap CI + TOST)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from stats_paired import boot_ci, wilcoxon_p

FOLDS_DIR = ROOT / "results" / "d1_folds"
mean = lambda v: sum(v) / len(v)


def boot_ci90(d, reps=10000):
    import random as _r
    _r.seed(0); nd = len(d)
    means = sorted(mean([d[_r.randrange(nd)] for _ in range(nd)]) for _ in range(reps))
    return means[int(0.05 * reps)], means[int(0.95 * reps)]


def paired(rpf, bpf, lower_is_better=False):
    sign = -1.0 if lower_is_better else 1.0
    d = [sign * (r - b) for r, b in zip(rpf, bpf)]
    lo, hi = boot_ci(d)
    return {"delta": round(mean(d), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "wilcoxon_p": round(wilcoxon_p(d), 4),
            "wins": f"{sum(1 for x in d if x > 0)}/{len(d)}"}


def tost(rpf, bpf, margin=0.02):
    d = [r - b for r, b in zip(rpf, bpf)]
    lo, hi = boot_ci90(d)
    return {"mean_delta": round(mean(d), 4), "ci90": [round(lo, 4), round(hi, 4)],
            "margin": margin, "equivalent": bool(lo > -margin and hi < margin)}


def main():
    folds = [json.loads(p.read_text())
             for p in sorted(FOLDS_DIR.glob("fold*.json"), key=lambda p: int(p.stem[4:]))]
    assert folds, "no fold json found"
    cfgs = list(folds[0]["soft"].keys())
    base_names = list(folds[0]["base"].keys())
    POOL = ["pst", "ppm", "ctw_leaf", "lz", "m1", "m2", "m3"]

    soft = {c: {k: [f["soft"][c][k] for f in folds] for k in ("acc1", "acc5", "mrr", "nll")} for c in cfgs}
    hard = {c: [f["hard"][c]["acc1"] for f in folds] for c in cfgs}
    base = {n: {k: [f["base"][n][k] for f in folds] for k in ("acc1", "acc5", "mrr", "nll")} for n in base_names}

    grid = [{"cfg": c, "acc1": round(mean(soft[c]["acc1"]), 4), "acc5": round(mean(soft[c]["acc5"]), 4),
             "mrr": round(mean(soft[c]["mrr"]), 4), "nll": round(mean(soft[c]["nll"]), 4),
             "hard_acc1": round(mean(hard[c]), 4)} for c in cfgs]
    baselines = {n: {k: round(mean(base[n][k]), 4) for k in ("acc1", "acc5", "mrr", "nll")} for n in base_names}
    best_cfg = max(cfgs, key=lambda c: mean(soft[c]["acc1"]))
    best_a1 = soft[best_cfg]["acc1"]; best_nll = soft[best_cfg]["nll"]

    fb = baselines["ctw_faithful_binary"]["acc1"]; lf = baselines["ctw_leaf"]["acc1"]; ff = baselines["ctw_faithful"]["acc1"]
    robust = {"cells_total": len(cfgs),
              "cells_soft_beats_ctw_leaf": sum(1 for c in cfgs if mean(soft[c]["acc1"]) > lf),
              "cells_soft_beats_ctw_faithful": sum(1 for c in cfgs if mean(soft[c]["acc1"]) > ff),
              "cells_soft_beats_ctw_faithful_binary": sum(1 for c in cfgs if mean(soft[c]["acc1"]) > fb)}

    fixed = max(POOL, key=lambda nm: baselines[nm]["acc1"])
    ps = {
        "acc1_vs_ctw_leaf": paired(best_a1, base["ctw_leaf"]["acc1"]),
        "acc1_vs_ctw_faithful": paired(best_a1, base["ctw_faithful"]["acc1"]),
        "acc1_vs_ctw_faithful_binary": paired(best_a1, base["ctw_faithful_binary"]["acc1"]),
        "nll_vs_ctw_leaf": paired(best_nll, base["ctw_leaf"]["nll"], lower_is_better=True),
        "nll_vs_ctw_faithful": paired(best_nll, base["ctw_faithful"]["nll"], lower_is_better=True),
        "acc1_vs_best_fixed": {"expert": fixed, **paired(best_a1, base[fixed]["acc1"]),
                               "tost": tost(best_a1, base[fixed]["acc1"])},
        "acc1_vs_m1": {**paired(best_a1, base["m1"]["acc1"]), "tost": tost(best_a1, base["m1"]["acc1"]),
                       "router_acc1_per_fold": [round(x, 4) for x in best_a1],
                       "m1_acc1_per_fold": [round(x, 4) for x in base["m1"]["acc1"]]},
    }
    res = {"corpus": "D1", "depth": folds[0]["depth"], "folds": len(folds),
           "total_test_events": sum(f["n"] for f in folds),
           "fold_test_uids": [f["test_uid"] for f in folds],
           "fold_wall_sec": [f["wall_sec"] for f in folds],
           "grid": grid, "baselines": baselines, "best_soft_cfg": best_cfg,
           "grid_robustness": robust, "paired": ps}
    out = ROOT / "results" / "route_grid_metrics_D1_full.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[D1 FULL] {res['folds']} folds, {res['total_test_events']} test events, best={best_cfg}")
    print(f"router acc@1 = {round(mean(best_a1),4)}")
    for k in ("acc1_vs_ctw_faithful_binary", "acc1_vs_ctw_leaf", "acc1_vs_best_fixed", "acc1_vs_m1"):
        v = ps[k]
        print(f"  {k}: d={v['delta']} p={v['wilcoxon_p']} wins={v['wins']} ci={v['ci95']}"
              + (f" expert={v['expert']}" if 'expert' in v else ""))
    print(f"  grid: beats fb {robust['cells_soft_beats_ctw_faithful_binary']}/{robust['cells_total']}, "
          f"leaf {robust['cells_soft_beats_ctw_leaf']}/{robust['cells_total']}")
    print(f"  fold wall (s): {res['fold_wall_sec']}")


if __name__ == "__main__":
    main()
