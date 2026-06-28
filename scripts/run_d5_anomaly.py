"""Anomaly detector on gigi sessions (sub-line d): reuse the validated Router-mix predictor and
flag the actions it finds most *surprising*. A realised action with high nll (low predicted
probability) given the behavioural history is an anomaly — an unusual step. Online accumulating:
the pool learns from each session in time order; high-nll positions are surfaced.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # ctrouter.py lives in scripts/

from data_d5_gigi import load_D5_event, load_D5_tool
from eval.metrics import nll_bits
from ctrouter import Router
from run_d5 import build

EXPERTS = ("pst", "ppm", "ctw", "lz")


def find_anomalies(dataset, depth=6, threshold_bits=8.0, warmup=100,
                   eta=0.5, alpha=0.0, experts=EXPERTS):
    """Run the Router-mix online over the time-ordered sessions; return the positions whose
    realised action has nll >= threshold_bits, sorted most-surprising first. The first `warmup`
    positions are NOT flagged (cold-start: everything is unseen, so high nll there is the model
    still learning the alphabet, not a genuine anomaly)."""
    items = sorted(dataset.items(), key=lambda kv: (kv[1][0], kv[0]))
    models = [build(name, depth) for name in experts]
    out, seen = [], 0
    for key, (_ts, seq) in items:
        router = Router(len(models), eta=eta, alpha=alpha)
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - depth): t]
            truth = seq[t + 1]
            dists = [m.predict(ctx) for m in models]
            mixed = router.predict_soft(dists)
            bits = nll_bits(mixed, truth)
            if seen >= warmup and bits >= threshold_bits:
                out.append({"session": key, "pos": t + 1, "context": ctx[-4:],
                            "action": truth, "nll_bits": round(bits, 2)})
            router.update(dists, truth)
            for m in models:
                m.update(ctx, truth)
            seen += 1
    return sorted(out, key=lambda a: -a["nll_bits"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["event", "tool"], default="tool")
    ap.add_argument("--root", default=os.path.expanduser("~/gigi"))
    ap.add_argument("--transcripts-root",
                    default=os.path.expanduser("~/.claude/projects/-Users-alessio-gigi"))
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--threshold-bits", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    data = (load_D5_event(a.root) if a.stream == "event"
            else load_D5_tool(a.transcripts_root))
    anomalies = find_anomalies(data, depth=a.depth, threshold_bits=a.threshold_bits,
                               warmup=a.warmup)
    out = a.out or f"results/anomalies_D5-{a.stream}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"stream": a.stream, "n_sessions": len(data),
                   "threshold_bits": a.threshold_bits, "n_anomalies": len(anomalies),
                   "anomalies": anomalies}, fh, indent=2)
    print(f"D5-{a.stream}: {len(anomalies)} anomalies (nll >= {a.threshold_bits} bits) "
          f"over {len(data)} sessions. Top {a.top} most surprising actions:")
    for x in anomalies[:a.top]:
        print(f"  {x['nll_bits']:6.1f} bits | {x['action']:10} after {x['context']} "
              f"| {x['session'][:18]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
