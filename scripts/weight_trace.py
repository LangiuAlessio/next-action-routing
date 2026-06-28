"""Router weight traces (Pilato review, 2026-06).

Records the Hedge/Fixed-Share weight vector at every step, at the corpus's
headline global cell, replicating ``route_grid_metrics.run_corpus`` (experts
update online, same skip rules).  Produces:

  * mean expert weight per corpus, averaged over LOUO folds (which experts
    the router actually trusts);
  * a per-step weight-evolution trace for one representative D3 test user
    (the longest stream), as a 7-curve figure.

Writes results/weight_trace.json and two figures into the paper figures dir.
Pure-Python + matplotlib (Agg).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from route_grid_metrics import _build_pool, POOL_NAMES  # noqa: E402
from ctrouter import Router                             # noqa: E402
from data_utils import load_D3, load_D4, leave_one_user_out  # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

DEPTH = 6
FIG_DIR = Path("/Users/alessio/gigi/ateneo/compact-trie-prediction-2026/figures")
DISPLAY = {"pst": "PST", "ppm": "PPM", "ctw_leaf": "CTW", "lz": "LZ",
           "m1": "M1", "m2": "M2", "m3": "M3"}
# headline global cell per corpus (best soft acc@1, from route_grid_metrics)
CELL = {"D3": (0.1, 0.05), "D4": (1.0, 0.05)}


def fold_steps(train, test):
    pool = _build_pool(train, DEPTH)
    recorded = []
    for seq in test:
        if len(seq) < 3:
            continue
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - DEPTH):t]
            truth = seq[t + 1]
            dists = [m.predict(ctx) for m in pool]
            recorded.append((dists, truth))
            for m in pool:
                m.update(ctx, truth)
    return recorded


def run_corpus(name, users):
    eta, alpha = CELL[name]
    K = len(POOL_NAMES)
    fold_means = []          # per-fold time-mean weight vector
    longest = None           # (uid, per-step weight matrix) for the figure
    for uid, train, test in leave_one_user_out(users):
        recorded = fold_steps(train, test)
        if not recorded:
            continue
        r = Router(K, eta=eta, alpha=alpha)
        wsum = [0.0] * K
        trace = []
        for dists, truth in recorded:
            w = r.weights()
            wsum = [wsum[i] + w[i] for i in range(K)]
            trace.append(list(w))
            r.update(dists, truth)
        n = len(recorded)
        fold_means.append([wsum[i] / n for i in range(K)])
        if longest is None or n > longest[2]:
            longest = (uid, trace, n)
    # mean over folds
    mean_w = [sum(fm[i] for fm in fold_means) / len(fold_means) for i in range(K)]
    return {"eta": eta, "alpha": alpha, "mean_weights": dict(zip(POOL_NAMES, mean_w)),
            "n_folds": len(fold_means)}, longest


def plot_evolution(name, longest, out_pdf):
    uid, trace, n = longest
    K = len(POOL_NAMES)
    xs = list(range(n))
    plt.figure(figsize=(7.2, 4.0))
    for i in range(K):
        ys = [trace[t][i] for t in range(n)]
        plt.plot(xs, ys, label=DISPLAY[POOL_NAMES[i]], linewidth=1.3)
    plt.xlabel("step (observed actions)")
    plt.ylabel("router weight (Hedge / Fixed-Share)")
    plt.title(f"{name} weight evolution, test user {uid}")
    plt.ylim(0, 1)
    plt.legend(ncol=4, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def plot_mean_bar(name, mean_weights, out_pdf):
    items = sorted(mean_weights.items(), key=lambda kv: -kv[1])
    labels = [DISPLAY[k] for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(5.4, 3.4))
    plt.bar(labels, vals, color="#3b6ea5")
    plt.ylabel("mean router weight")
    plt.title(f"{name}: mean expert weight over {len(mean_weights)} experts, LOUO folds")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main():
    FIG_DIR.mkdir(exist_ok=True)
    res = {}
    for name, loader in (("D3", load_D3), ("D4", load_D4)):
        print(f"=== {name} ===", flush=True)
        users = loader()
        summary, longest = run_corpus(name, users)
        res[name] = summary
        mw = summary["mean_weights"]
        for k in sorted(mw, key=lambda x: -mw[x]):
            print(f"  {DISPLAY[k]:4s} {mw[k]:.4f}", flush=True)
        plot_mean_bar(name, mw, FIG_DIR / f"router-weight-mean-{name.lower()}.pdf")
        if name == "D3":
            plot_evolution(name, longest, FIG_DIR / "router-weight-evolution-d3.pdf")
            print(f"  evolution figure: user {longest[0]}, {longest[2]} steps", flush=True)
    out_path = ROOT / "results" / "weight_trace.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}\nfigures in {FIG_DIR}")


if __name__ == "__main__":
    main()
