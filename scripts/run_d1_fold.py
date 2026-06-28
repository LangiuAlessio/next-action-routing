"""Run ONE LOUO fold of D1 (full, depth 6) and save per-fold metrics. Parallelize across the 9
folds to make full D1 feasible (~slowest fold instead of ~38h serial). Same per-fold computation
as route_grid_metrics.run_corpus, factored to one fold. Aggregate with aggregate_d1.py."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import route_grid_metrics as R
from data_utils import load_D1, leave_one_user_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "d1_folds"))
    a = ap.parse_args()
    splits = list(leave_one_user_out(load_D1()))
    test_uid, train, test = splits[a.fold]
    t0 = time.time()
    pool = R._build_pool(train, a.depth)
    faithful = R.CTWFaithful(depth=a.depth); faithful.fit(train)
    fb = R.CTWFaithfulBinary(depth=a.depth); fb.fit(train)

    recorded = []
    for seq in test:
        if len(seq) < 3:
            continue
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - a.depth):t]; truth = seq[t + 1]
            dists = [m.predict(ctx) for m in pool]
            recorded.append((dists, faithful.predict(ctx), fb.predict(ctx), truth, len(ctx)))
            for m in pool:
                m.update(ctx, truth)
            faithful.update(ctx, truth); fb.update(ctx, truth)
    n = len(recorded)

    base = {}
    for bi, name in enumerate(R.POOL_NAMES):
        acc = [0.0, 0.0, 0.0, 0.0]
        for dists, _f, _fb, truth, _h in recorded:
            v = R._metrics_from_dist(dists[bi], truth)
            for j in range(4):
                acc[j] += v[j]
        base[name] = {"acc1": acc[0]/n, "acc5": acc[1]/n, "mrr": acc[2]/n, "nll": acc[3]/n}
    for idx, key in ((1, "ctw_faithful"), (2, "ctw_faithful_binary")):
        acc = [0.0, 0.0, 0.0, 0.0]
        for rec in recorded:
            v = R._metrics_from_dist(rec[idx], rec[3])
            for j in range(4):
                acc[j] += v[j]
        base[key] = {"acc1": acc[0]/n, "acc5": acc[1]/n, "mrr": acc[2]/n, "nll": acc[3]/n}

    soft = {}; hard = {}
    for sc in R.SCOPES:
        for e in R.ETAS:
            for al in R.ALPHAS:
                cfg = f"{sc}|eta{e}|a{al}"
                if sc == "g":
                    routers = {0: R.Router(len(R.POOL_NAMES), eta=e, alpha=al)}
                    def pick(_h, routers=routers):
                        return routers[0]
                else:
                    routers = {b: R.Router(len(R.POOL_NAMES), eta=e, alpha=al) for b in range(4)}
                    def pick(h, routers=routers):
                        return routers[R._bucket(h)]
                s = [0.0, 0.0, 0.0, 0.0]; h1 = 0.0
                for dists, _f, _fb, truth, hlen in recorded:
                    r = pick(hlen)
                    v = R._metrics_from_dist(r.predict_soft(dists), truth)
                    for j in range(4):
                        s[j] += v[j]
                    h1 += 1.0 if R.acc_at_k(R._top(r.predict_hard(dists)), truth, 1) else 0.0
                    r.update(dists, truth)
                soft[cfg] = {"acc1": s[0]/n, "acc5": s[1]/n, "mrr": s[2]/n, "nll": s[3]/n}
                hard[cfg] = {"acc1": h1/n}

    out = {"fold": a.fold, "test_uid": test_uid, "n": n, "depth": a.depth,
           "base": base, "soft": soft, "hard": hard, "wall_sec": round(time.time() - t0, 1)}
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    Path(a.outdir, f"fold{a.fold}.json").write_text(json.dumps(out))
    print(f"fold {a.fold} ({test_uid}) done n={n} in {out['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
