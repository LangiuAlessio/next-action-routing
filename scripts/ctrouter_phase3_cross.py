"""Phase 3 — cross-corpus re-run of the ctrouter probe on D3 + D4.

Reuses the Hedge+Fixed-Share Router from `scripts/ctrouter.py`, the
expert pool from `scripts/markov_k.py` + bench `_build_model`, and the
LOUO loaders from `src/data_utils.py`. The probe driver is the same
shape as `scripts/ctrouter_probe.py` (existing D3-only GO probe with
soft_best +0.036 / wins 11/12) but extended to:

  - run D3 (re-verify reproducibility of the previous GO result)
  - run D4 (new corpus; CTW feasible since |Σ|=7)
  - retain per-fold accuracies for the best soft/hard router cfgs so
    that paired stats (bootstrap CI, Wilcoxon, fold-consistency) can be
    reported per corpus, plus Holm correction across the 2 corpora.

D1 remains scope-cut: CTW is infeasible there (|Σ|≈400, >25 min/fold),
so the C1 bar "router beats static CTW" is undefined. The existing
ctrouter_probe.py memoised that decision; we keep it.

C1 framing reminder (paper rescope, DR-CTS-skip): the "static CTW"
incumbent here is the bench's depth-D leaf-KT VOMM CTW variant; the
paper §2 will declare faithful Willems CTW + CTS Veness 2012 as future
work. So C1 strictly reads "router beats depth-D leaf-KT VOMM family"
on the corpora reported.

Output: results/ctrouter_phase3_cross.json with per-corpus per-fold
arrays + per-corpus best-cfg pick + cross-corpus Holm-corrected p.
"""
from __future__ import annotations
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_utils import load_D3, load_D4, leave_one_user_out
from run_benchmark import _build_model
from markov_k import MarkovK
from ctrouter import Router
from stats_paired import boot_ci, wilcoxon_p

DEPTH = 6
ETAS = [0.1, 0.5, 1.0]
ALPHAS = [0.0, 0.01, 0.05]
EXPERTS = ["pst", "ppm", "ctw", "lz", "mk1", "mk2", "mk3"]


def make_pool():
    pool = []
    for name in EXPERTS:
        if name.startswith("mk"):
            pool.append(MarkovK(k=int(name[2:])))
        else:
            pool.append(_build_model(name, DEPTH))
    return pool


def ctx_bucket(ctx):
    L = min(len(ctx), 6)
    return 0 if L == 0 else (1 if L <= 2 else (2 if L <= 4 else 3))


def run_corpus(loader, label: str):
    """Returns per-fold dicts identical to ctrouter_probe.py, plus the
    raw per-fold (n, ctw_ok, soft_ok, hard_ok) so we can compute paired
    deltas on absolute accuracies (not pre-averaged)."""
    folds = []
    for u, tr, te in leave_one_user_out(loader()):
        pool = make_pool()
        for m in pool:
            m.fit(tr)
        ctw_idx = EXPERTS.index("ctw")
        cfgs = {}
        for e in ETAS:
            for a in ALPHAS:
                cfgs[("g", e, a)] = Router(len(pool), e, a)
                cfgs[("c", e, a)] = {b: Router(len(pool), e, a) for b in range(4)}
        n = 0
        exp_ok = [0] * len(pool)
        ctw_ok = 0
        any_ok = 0
        soft_ok = {k: 0 for k in cfgs}
        hard_ok = {k: 0 for k in cfgs}
        for seq in te:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - DEPTH):t]
                truth = seq[t + 1]
                dists = [m.predict(ctx) for m in pool]
                tops = [max(d, key=d.get) if d else None for d in dists]
                for i, tp in enumerate(tops):
                    exp_ok[i] += (tp == truth)
                ctw_ok += (tops[ctw_idx] == truth)
                any_ok += any(tp == truth for tp in tops)
                b = ctx_bucket(ctx)
                for key, r in cfgs.items():
                    rt = r[b] if key[0] == "c" else r
                    ps = rt.predict_soft(dists)
                    ph = rt.predict_hard(dists)
                    soft_ok[key] += (bool(ps) and max(ps, key=ps.get) == truth)
                    hard_ok[key] += (bool(ph) and max(ph, key=ph.get) == truth)
                    rt.update(dists, truth)
                for m in pool:
                    m.update(ctx, truth)
                n += 1
        if n == 0:
            continue
        folds.append(dict(
            user=u, n=n,
            experts={EXPERTS[i]: exp_ok[i] / n for i in range(len(pool))},
            ctw=ctw_ok / n,
            any_oracle=any_ok / n,
            soft={f"{k[0]}|eta{k[1]}|a{k[2]}": v / n for k, v in soft_ok.items()},
            hard={f"{k[0]}|eta{k[1]}|a{k[2]}": v / n for k, v in hard_ok.items()},
        ))
        print(f"  [{label}] fold {u}: ctw={ctw_ok/n:.3f} "
              f"best_soft={max(v/n for v in soft_ok.values()):.3f} n={n}")
    return folds


def best_cfg_summary(folds, side: str):
    """Pick the cfg maximising mean accuracy across folds, then return
    per-fold accuracies + delta-vs-ctw + Wilcoxon p + bootstrap CI of
    the mean delta + fold-consistency count."""
    keys = list(folds[0][side].keys())
    means = {k: st.mean(f[side][k] for f in folds) for k in keys}
    best_key = max(means, key=means.get)
    per_fold_router = [f[side][best_key] for f in folds]
    per_fold_ctw = [f["ctw"] for f in folds]
    per_fold_delta = [r - c for r, c in zip(per_fold_router, per_fold_ctw)]
    wilc_p = wilcoxon_p(per_fold_delta)
    ci_lo, ci_hi = boot_ci(per_fold_delta)
    wins = sum(1 for d in per_fold_delta if d > 0)
    return dict(
        cfg=best_key,
        acc_router=round(st.mean(per_fold_router), 4),
        acc_ctw=round(st.mean(per_fold_ctw), 4),
        mean_delta=round(st.mean(per_fold_delta), 4),
        delta_ci95=[round(ci_lo, 4), round(ci_hi, 4)],
        wilcoxon_p=round(wilc_p, 4),
        wins=f"{wins}/{len(folds)}",
        per_fold_router=[round(x, 4) for x in per_fold_router],
        per_fold_ctw=[round(x, 4) for x in per_fold_ctw],
        per_fold_delta=[round(x, 4) for x in per_fold_delta],
    )


def holm_correct(p_values: dict) -> dict:
    """Holm step-down correction. Returns {key: corrected_p}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, p * (n - i)))
        out[k] = round(adj, 4)
        prev = adj
    return out


def main():
    random.seed(0)
    results = {}
    raw_p_soft = {}
    raw_p_hard = {}
    for label, loader in [("D3", load_D3), ("D4", load_D4)]:
        print(f"\n=== {label} ===")
        folds = run_corpus(loader, label)
        soft = best_cfg_summary(folds, "soft")
        hard = best_cfg_summary(folds, "hard")
        # extra context for the paper
        keys = list(folds[0]["experts"].keys())
        best_expert = max(
            keys, key=lambda e: st.mean(f["experts"][e] for f in folds))
        results[label] = dict(
            folds=len(folds),
            best_fixed_expert=dict(
                name=best_expert,
                acc=round(st.mean(f["experts"][best_expert] for f in folds), 4)),
            any_oracle=round(st.mean(f["any_oracle"] for f in folds), 4),
            router_soft=soft,
            router_hard=hard,
        )
        raw_p_soft[label] = soft["wilcoxon_p"]
        raw_p_hard[label] = hard["wilcoxon_p"]
        print(f"  [{label}] soft Δ={soft['mean_delta']} "
              f"CI95={soft['delta_ci95']} wilc_p={soft['wilcoxon_p']} "
              f"wins={soft['wins']}")
        print(f"  [{label}] hard Δ={hard['mean_delta']} "
              f"CI95={hard['delta_ci95']} wilc_p={hard['wilcoxon_p']} "
              f"wins={hard['wins']}")

    holm_soft = holm_correct(raw_p_soft)
    holm_hard = holm_correct(raw_p_hard)
    for label in results:
        results[label]["router_soft"]["wilcoxon_p_holm"] = holm_soft[label]
        results[label]["router_hard"]["wilcoxon_p_holm"] = holm_hard[label]

    out = ROOT / "results" / "ctrouter_phase3_cross.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[write] {out}")

    print("\n=== CROSS-CORPUS SUMMARY (router-soft vs static-CTW) ===")
    for label, r in results.items():
        s = r["router_soft"]
        print(f"  {label}: Δ={s['mean_delta']:+.4f}  "
              f"CI95={s['delta_ci95']}  wilc_p={s['wilcoxon_p']}  "
              f"holm={s['wilcoxon_p_holm']}  wins={s['wins']}  cfg={s['cfg']}")


if __name__ == "__main__":
    main()
