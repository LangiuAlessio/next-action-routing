"""Run the full router-vs-CTW grid on D1 (Unix shell, |Sigma|=1514) -- the paper's scope-cut corpus.
Uses the SAME run_corpus as D3/D4 (no shortcuts to the method). --cap truncates per-user events
(for timing probes / disclosed-cost runs); --cap 0 = full D1. Honest: any cap is reported.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from data_utils import load_D1
import route_grid_metrics as R


def cap_users(users, cap):
    if not cap:
        return users
    out = {}
    for uid, sessions in users.items():
        kept, tot = [], 0
        for s in sessions:
            if tot >= cap:
                break
            kept.append(s); tot += len(s)
        out[uid] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--cap", type=int, default=0, help="max events/user (0=full)")
    ap.add_argument("--out", default=str(ROOT / "results" / "route_grid_metrics_D1.json"))
    a = ap.parse_args()
    users = cap_users(load_D1(), a.cap)
    tot = sum(sum(len(s) for s in ss) for ss in users.values())
    print(f"[D1] {len(users)} users, {tot} events, depth={a.depth}, cap={a.cap}", flush=True)
    t0 = time.time()
    res = R.run_corpus(users, depth=a.depth)
    res["corpus"] = "D1"; res["cap"] = a.cap; res["wall_sec"] = round(time.time() - t0, 1)
    res["total_events"] = tot
    Path(a.out).write_text(json.dumps(res, indent=2))
    ps = res.get("paired_stats", {})
    print(f"[D1] done {res['folds']} folds in {res['wall_sec']}s  best={res['best_soft_cfg']}", flush=True)
    for k in ("acc1_vs_ctw_faithful_binary", "acc1_vs_ctw_leaf", "acc1_vs_m1"):
        if k in ps:
            print(f"  {k}: delta={ps[k]['delta']} p={ps[k].get('wilcoxon_p')} wins={ps[k].get('wins')} ci={ps[k].get('ci95')}", flush=True)
    gr = res.get("grid_robustness", {})
    print(f"  grid: beats faithful_bin {gr.get('cells_soft_beats_ctw_faithful_binary')}/{gr.get('cells_total')}, "
          f"leaf {gr.get('cells_soft_beats_ctw_leaf')}/{gr.get('cells_total')}", flush=True)


if __name__ == "__main__":
    main()
