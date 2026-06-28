"""Router-mix on gigi D5-event: combine the PST-family pool per position with the online
Hedge + Fixed-Share Router (ctrouter.py). Soft mixture and hard top-1 (MoE-faithful) read-out.
Fresh Router per session (per-session reset). Does the mix beat the best single predictor?
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # ctrouter.py lives in scripts/

from data_d5_gigi import load_D5_event, load_D5_tool, temporal_split
from eval.metrics import acc_at_k, nll_bits
from ctrouter import Router
from run_d5 import build  # reuse the predictor factory

EXPERTS = ("pst", "ppm", "ctw", "lz")


def run_router(splits, depth=6, k=5, eta=0.5, alpha=0.0, readout="soft", experts=EXPERTS):
    acc1, acc5, nll, lat = [], [], [], []
    for _key, train, test in splits:
        models = [build(name, depth) for name in experts]
        for m in models:
            m.fit(train)
        for seq in test:
            router = Router(len(models), eta=eta, alpha=alpha)  # fresh per session
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - depth): t]
                truth = seq[t + 1]
                t0 = time.perf_counter()
                dists = [m.predict(ctx) for m in models]
                mixed = router.predict_soft(dists) if readout == "soft" else router.predict_hard(dists)
                lat.append((time.perf_counter() - t0) * 1000)
                top = sorted(mixed.items(), key=lambda x: -x[1])[:k]
                acc1.append(acc_at_k(top, truth, k=1))
                acc5.append(acc_at_k(top, truth, k=k))
                nll.append(nll_bits(mixed, truth))
                router.update(dists, truth)
                for m in models:
                    m.update(ctx, truth)
    n = max(len(acc1), 1)
    return {"model": f"router-{readout}", "n_steps": len(acc1),
            "acc_at_1": round(sum(acc1) / n, 4), "acc_at_5": round(sum(acc5) / n, 4),
            "nll_bits": round(sum(nll) / n, 4),
            "latency_ms": round(sum(lat) / max(len(lat), 1), 5)}


def run_router_online(dataset, depth=6, k=5, eta=0.5, alpha=0.0, readout="soft", experts=EXPERTS):
    """Online accumulating Router-mix (O(total positions)): the expert pool accumulates across
    sessions in time order; a fresh Router per session mixes them. Scales to thousands of sessions."""
    items = sorted(dataset.items(), key=lambda kv: (kv[1][0], kv[0]))
    models = [build(name, depth) for name in experts]
    acc1, acc5, nll, lat = [], [], [], []
    for _key, (_ts, seq) in items:
        router = Router(len(models), eta=eta, alpha=alpha)  # fresh per session
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - depth): t]
            truth = seq[t + 1]
            t0 = time.perf_counter()
            dists = [m.predict(ctx) for m in models]
            mixed = router.predict_soft(dists) if readout == "soft" else router.predict_hard(dists)
            lat.append((time.perf_counter() - t0) * 1000)
            top = sorted(mixed.items(), key=lambda x: -x[1])[:k]
            acc1.append(acc_at_k(top, truth, k=1))
            acc5.append(acc_at_k(top, truth, k=k))
            nll.append(nll_bits(mixed, truth))
            router.update(dists, truth)
            for m in models:
                m.update(ctx, truth)
    n = max(len(acc1), 1)
    return {"model": f"router-{readout}", "n_steps": len(acc1),
            "acc_at_1": round(sum(acc1) / n, 4), "acc_at_5": round(sum(acc5) / n, 4),
            "nll_bits": round(sum(nll) / n, 4),
            "latency_ms": round(sum(lat) / max(len(lat), 1), 5)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["event", "tool"], default="event")
    ap.add_argument("--root", default=os.path.expanduser("~/gigi"))
    ap.add_argument("--transcripts-root",
                    default=os.path.expanduser("~/.claude/projects/-Users-alessio-gigi"))
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    data = (load_D5_event(a.root) if a.stream == "event"
            else load_D5_tool(a.transcripts_root))
    out = a.out or f"results/benchmark_D5-{a.stream}_router.json"
    rows = [run_router_online(data, depth=a.depth, eta=a.eta, alpha=a.alpha, readout=ro)
            for ro in ("soft", "hard")]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"stream": a.stream, "n_sessions": len(data), "eta": a.eta,
                   "alpha": a.alpha, "models": rows}, fh, indent=2)
    print(f"D5-{a.stream} Router-mix ({len(data)} sessions, eta={a.eta}, alpha={a.alpha}):")
    print(f"  {'model':12} {'acc@1':>6} {'acc@5':>6} {'nll':>6} {'lat_ms':>9}")
    for r in rows:
        print(f"  {r['model']:12} {r['acc_at_1']:>6.3f} {r['acc_at_5']:>6.3f} "
              f"{r['nll_bits']:>6.2f} {r['latency_ms']:>9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
