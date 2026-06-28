"""
Full sensitivity grid + probabilistic metrics for the online expert router.

Answers Pilato review #2/#4/#5/#6 in one pass:
  #5  full 18-cell (eta x alpha x {global, context-bucketed}) sensitivity grid;
  #2  reporting all cells makes config selection non-cherry-picked — the router
      beats CTW across (almost) the whole grid, so the headline does not depend
      on which cell is chosen;
  #4/#6  every cell and baseline is scored on acc@1, acc@5, MRR and NLL(bits),
      so we can say whether the router redistributes probability mass (NLL) or
      only flips the top-1 (acc@1).

The static baselines (leaf-KT CTW, faithful Willems CTW, the fixed experts)
are scored from the same per-step expert distributions, so the comparison is
exactly paired by fold and step.

Experts update online identically regardless of router config, so each fold is
streamed once (recording per-expert distributions + truth + context length),
then the 18 router configs are replayed over the recorded distributions.

Usage: python3 scripts/route_grid_metrics.py [--depth 6]
Writes results/route_grid_metrics.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.pst import PST                      # noqa: E402
from baselines.ppm import PPM                      # noqa: E402
from baselines.ctw import CTW                      # noqa: E402
from baselines.ctw_faithful import CTWFaithful, CTWFaithfulBinary  # noqa: E402
from baselines.lz_pred import LZPredictor          # noqa: E402
from markov_k import MarkovK                        # noqa: E402
from ctrouter import Router                         # noqa: E402
from data_utils import load_D3, load_D4, leave_one_user_out  # noqa: E402
from eval.metrics import acc_at_k, nll_bits, reciprocal_rank  # noqa: E402
from stats_paired import boot_ci, wilcoxon_p        # noqa: E402

ETAS = [0.1, 0.5, 1.0]
ALPHAS = [0.0, 0.01, 0.05]
SCOPES = ["g", "c"]          # global / context-bucketed
POOL_NAMES = ["pst", "ppm", "ctw_leaf", "lz", "m1", "m2", "m3"]


def _bucket(hlen: int) -> int:
    if hlen == 0:
        return 0
    if hlen <= 2:
        return 1
    if hlen <= 4:
        return 2
    return 3


def _top(dist: dict) -> list[tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: -kv[1])


def _build_pool(train, depth):
    pool = [
        PST(max_depth=depth, min_count=2, min_gain=0.01),
        PPM(max_order=depth),
        CTW(depth=depth),
        LZPredictor(),
        MarkovK(1), MarkovK(2), MarkovK(3),
    ]
    for m in pool:
        m.fit(train)
    return pool


def _metrics_from_dist(dist, truth):
    top = _top(dist)
    return (
        1.0 if acc_at_k(top, truth, 1) else 0.0,
        1.0 if acc_at_k(top, truth, 5) else 0.0,
        reciprocal_rank(top, truth),
        nll_bits(dist, truth),
    )


def _mean(xs):
    return sum(xs) / max(len(xs), 1)


def run_corpus(users, depth):
    splits = leave_one_user_out(users)

    # per-fold metric accumulators
    grid_cfgs = [f"{sc}|eta{e}|a{a}" for sc in SCOPES for e in ETAS for a in ALPHAS]
    soft = {c: {"acc1": [], "acc5": [], "mrr": [], "nll": []} for c in grid_cfgs}
    hard = {c: {"acc1": []} for c in grid_cfgs}
    base_names = POOL_NAMES + ["ctw_faithful", "ctw_faithful_binary"]
    base = {n: {"acc1": [], "acc5": [], "mrr": [], "nll": []} for n in base_names}

    for test_uid, train, test in splits:
        pool = _build_pool(train, depth)
        faithful = CTWFaithful(depth=depth)
        faithful.fit(train)
        faithful_bin = CTWFaithfulBinary(depth=depth)
        faithful_bin.fit(train)

        # stream once, record per-expert distributions
        recorded = []  # (pool_dists, faithful_dist, faithful_bin_dist, truth, hlen)
        for seq in test:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - depth):t]
                truth = seq[t + 1]
                dists = [m.predict(ctx) for m in pool]
                fdist = faithful.predict(ctx)
                fbdist = faithful_bin.predict(ctx)
                recorded.append((dists, fdist, fbdist, truth, len(ctx)))
                for m in pool:
                    m.update(ctx, truth)
                faithful.update(ctx, truth)
                faithful_bin.update(ctx, truth)

        if not recorded:
            continue
        n = len(recorded)

        # ---- static baselines (each pool expert + faithful + faithful-bin) ----
        for bi, name in enumerate(POOL_NAMES):
            a1 = a5 = mr = nl = 0.0
            for dists, _f, _fb, truth, _h in recorded:
                m1_, m5_, mr_, nl_ = _metrics_from_dist(dists[bi], truth)
                a1 += m1_; a5 += m5_; mr += mr_; nl += nl_
            base[name]["acc1"].append(a1 / n); base[name]["acc5"].append(a5 / n)
            base[name]["mrr"].append(mr / n); base[name]["nll"].append(nl / n)
        for idx, key in ((1, "ctw_faithful"), (2, "ctw_faithful_binary")):
            a1 = a5 = mr = nl = 0.0
            for rec in recorded:
                m1_, m5_, mr_, nl_ = _metrics_from_dist(rec[idx], rec[3])
                a1 += m1_; a5 += m5_; mr += mr_; nl += nl_
            base[key]["acc1"].append(a1 / n); base[key]["acc5"].append(a5 / n)
            base[key]["mrr"].append(mr / n); base[key]["nll"].append(nl / n)

        # ---- replay each of the 18 router configs over recorded dists ----
        for sc in SCOPES:
            for e in ETAS:
                for a in ALPHAS:
                    cfg = f"{sc}|eta{e}|a{a}"
                    if sc == "g":
                        routers = {0: Router(len(POOL_NAMES), eta=e, alpha=a)}

                        def pick(_h):
                            return routers[0]
                    else:
                        routers = {b: Router(len(POOL_NAMES), eta=e, alpha=a) for b in range(4)}

                        def pick(h):
                            return routers[_bucket(h)]

                    s1 = s5 = smr = snl = h1 = 0.0
                    for dists, _f, _fb, truth, hlen in recorded:
                        r = pick(hlen)
                        soft_dist = r.predict_soft(dists)
                        m1_, m5_, mr_, nl_ = _metrics_from_dist(soft_dist, truth)
                        s1 += m1_; s5 += m5_; smr += mr_; snl += nl_
                        hard_dist = r.predict_hard(dists)
                        htop = _top(hard_dist)
                        h1 += 1.0 if acc_at_k(htop, truth, 1) else 0.0
                        r.update(dists, truth)
                    soft[cfg]["acc1"].append(s1 / n); soft[cfg]["acc5"].append(s5 / n)
                    soft[cfg]["mrr"].append(smr / n); soft[cfg]["nll"].append(snl / n)
                    hard[cfg]["acc1"].append(h1 / n)

    # ---- aggregate ----
    def agg(d):
        return {k: round(_mean(v), 4) for k, v in d.items()}

    grid = []
    for c in grid_cfgs:
        grid.append({
            "cfg": c,
            "acc1": round(_mean(soft[c]["acc1"]), 4),
            "acc5": round(_mean(soft[c]["acc5"]), 4),
            "mrr": round(_mean(soft[c]["mrr"]), 4),
            "nll": round(_mean(soft[c]["nll"]), 4),
            "hard_acc1": round(_mean(hard[c]["acc1"]), 4),
        })
    baselines = {n: agg(base[n]) for n in base_names}

    # best soft cell by mean acc@1
    best = max(grid, key=lambda g: g["acc1"])
    best_cfg = best["cfg"]
    best_acc1_pf = soft[best_cfg]["acc1"]
    best_nll_pf = soft[best_cfg]["nll"]

    # grid robustness: in how many of the 18 cells does soft router beat the
    # two CTW baselines on mean acc@1?
    ctw_leaf_acc1 = baselines["ctw_leaf"]["acc1"]
    ctw_f_acc1 = baselines["ctw_faithful"]["acc1"]
    ctw_fb_acc1 = baselines["ctw_faithful_binary"]["acc1"]
    cells_beat_leaf = sum(1 for g in grid if g["acc1"] > ctw_leaf_acc1)
    cells_beat_faithful = sum(1 for g in grid if g["acc1"] > ctw_f_acc1)
    cells_beat_faithful_bin = sum(1 for g in grid if g["acc1"] > ctw_fb_acc1)

    def paired(metric_pf_router, metric_pf_base, lower_is_better=False):
        sign = -1.0 if lower_is_better else 1.0
        deltas = [sign * (r - b) for r, b in zip(metric_pf_router, metric_pf_base)]
        lo, hi = boot_ci(deltas)
        return {
            "delta": round(_mean(deltas), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "wilcoxon_p": round(wilcoxon_p(deltas), 4),
            "wins": f"{sum(1 for x in deltas if x > 0)}/{len(deltas)}",
        }

    def boot_ci90(d, reps=10000):
        import random as _r
        _r.seed(0)
        nd = len(d)
        means = sorted(_mean([d[_r.randrange(nd)] for _ in range(nd)]) for _ in range(reps))
        return means[int(0.05 * reps)], means[int(0.95 * reps)]

    def tost(router_pf, base_pf, margin=0.02):
        """Equivalence by the (1-2alpha) CI rule: equivalent at alpha=0.05 iff
        the 90% bootstrap CI of the mean paired difference lies within +/-margin."""
        deltas = [r - b for r, b in zip(router_pf, base_pf)]
        lo, hi = boot_ci90(deltas)
        return {
            "mean_delta": round(_mean(deltas), 4),
            "ci90": [round(lo, 4), round(hi, 4)],
            "margin": margin,
            "equivalent": bool(lo > -margin and hi < margin),
        }

    paired_stats = {
        "acc1_vs_ctw_leaf": paired(best_acc1_pf, base["ctw_leaf"]["acc1"]),
        "acc1_vs_ctw_faithful": paired(best_acc1_pf, base["ctw_faithful"]["acc1"]),
        "acc1_vs_ctw_faithful_binary": paired(best_acc1_pf, base["ctw_faithful_binary"]["acc1"]),
        "nll_vs_ctw_leaf": paired(best_nll_pf, base["ctw_leaf"]["nll"], lower_is_better=True),
        "nll_vs_ctw_faithful": paired(best_nll_pf, base["ctw_faithful"]["nll"], lower_is_better=True),
    }
    # vs best fixed expert (by acc1): paired test + TOST equivalence
    fixed = max(POOL_NAMES, key=lambda nm: baselines[nm]["acc1"])
    paired_stats["acc1_vs_best_fixed"] = {
        "expert": fixed,
        **paired(best_acc1_pf, base[fixed]["acc1"]),
        "tost": tost(best_acc1_pf, base[fixed]["acc1"]),
    }
    # also explicit vs Markov-1 (the deployer's pragmatic single-model choice)
    paired_stats["acc1_vs_m1"] = {
        **paired(best_acc1_pf, base["m1"]["acc1"]),
        "tost": tost(best_acc1_pf, base["m1"]["acc1"]),
        "router_acc1_per_fold": [round(x, 4) for x in best_acc1_pf],
        "m1_acc1_per_fold": [round(x, 4) for x in base["m1"]["acc1"]],
    }

    return {
        "folds": len(best_acc1_pf),
        "depth": depth,
        "grid": grid,
        "baselines": baselines,
        "best_soft_cfg": best_cfg,
        "grid_robustness": {
            "cells_total": len(grid),
            "cells_soft_beats_ctw_leaf": cells_beat_leaf,
            "cells_soft_beats_ctw_faithful": cells_beat_faithful,
            "cells_soft_beats_ctw_faithful_binary": cells_beat_faithful_bin,
        },
        "paired": paired_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    out = {}
    for name, loader in (("D3", load_D3), ("D4", load_D4)):
        print(f"=== {name} ===", flush=True)
        users = loader()
        res = run_corpus(users, args.depth)
        out[name] = res
        print(f"  best soft {res['best_soft_cfg']}: "
              f"acc1={dict((g['cfg'], g['acc1']) for g in res['grid'])[res['best_soft_cfg']]}, "
              f"CTW leaf={res['baselines']['ctw_leaf']['acc1']}, "
              f"CTW faithful={res['baselines']['ctw_faithful']['acc1']}", flush=True)
        print(f"  grid robustness: beats leaf in "
              f"{res['grid_robustness']['cells_soft_beats_ctw_leaf']}/18, "
              f"beats faithful in {res['grid_robustness']['cells_soft_beats_ctw_faithful']}/18",
              flush=True)

    out_path = ROOT / "results" / "route_grid_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
