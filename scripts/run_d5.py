"""Temporal/online next-action benchmark on gigi D5-event.

Per session (ordered by time): fit predictors on all prior sessions, then predict each
next action online (predict -> score -> update) within the session. Reuses the bench's
predictors and metrics. Stdlib-only, local, no LLM.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_d5_gigi import load_D5_event, load_D5_tool, temporal_split
from baselines.markov0 import Markov0
from baselines.pst import PST
from baselines.ppm import PPM
from baselines.ctw import CTW
from baselines.lz_pred import LZPredictor
from eval.metrics import acc_at_k, nll_bits, latency_ms


def build(name, depth):
    return {
        "markov0": lambda: Markov0(),
        "pst": lambda: PST(max_depth=depth, min_count=2, min_gain=0.01),
        "ppm": lambda: PPM(max_order=depth),
        "ctw": lambda: CTW(depth=depth),
        "lz": lambda: LZPredictor(),
    }[name]()


def run_model(name, splits, depth=6, k=5):
    acc1, acc5, nll, lat = [], [], [], []
    for _key, train, test in splits:
        m = build(name, depth)
        m.fit(train)
        for seq in test:
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - depth): t]
                truth = seq[t + 1]
                top, ms = latency_ms(m.predict_top, ctx, k=k)
                lat.append(ms)
                acc1.append(acc_at_k(top, truth, k=1))
                acc5.append(acc_at_k(top, truth, k=k))
                nll.append(nll_bits(m.predict(ctx), truth))
                m.update(ctx, truth)
    n = max(len(acc1), 1)
    return {"model": name, "n_steps": len(acc1),
            "acc_at_1": round(sum(acc1) / n, 4), "acc_at_5": round(sum(acc5) / n, 4),
            "nll_bits": round(sum(nll) / n, 4),
            "latency_ms": round(sum(lat) / max(len(lat), 1), 5)}


def run_model_online(name, dataset, depth=6, k=5):
    """Truly-online accumulating protocol (O(total positions)): one model accumulates across
    sessions in time order, predicting each next action then updating. For count-based predictors
    this equals the per-session refit-on-history protocol but scales to thousands of sessions."""
    items = sorted(dataset.items(), key=lambda kv: (kv[1][0], kv[0]))
    m = build(name, depth)
    acc1, acc5, nll, lat = [], [], [], []
    for _key, (_ts, seq) in items:
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - depth): t]
            truth = seq[t + 1]
            top, ms = latency_ms(m.predict_top, ctx, k=k)
            lat.append(ms)
            acc1.append(acc_at_k(top, truth, k=1))
            acc5.append(acc_at_k(top, truth, k=k))
            nll.append(nll_bits(m.predict(ctx), truth))
            m.update(ctx, truth)
    n = max(len(acc1), 1)
    return {"model": name, "n_steps": len(acc1),
            "acc_at_1": round(sum(acc1) / n, 4), "acc_at_5": round(sum(acc5) / n, 4),
            "nll_bits": round(sum(nll) / n, 4),
            "latency_ms": round(sum(lat) / max(len(lat), 1), 5)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["event", "tool"], default="event")
    ap.add_argument("--root", default=os.path.expanduser("~/gigi"))
    ap.add_argument("--transcripts-root",
                    default=os.path.expanduser(os.environ.get("D5_TRANSCRIPTS_ROOT", "~/.claude/projects")),
                    help="Claude Code project directory whose transcripts form D5 (env D5_TRANSCRIPTS_ROOT)")
    ap.add_argument("--models", nargs="+", default=["markov0", "pst", "ppm", "ctw", "lz"])
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    data = (load_D5_event(a.root) if a.stream == "event"
            else load_D5_tool(a.transcripts_root))
    out = a.out or f"results/benchmark_D5-{a.stream}.json"
    rows = [run_model_online(m, data, depth=a.depth) for m in a.models]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"stream": a.stream, "n_sessions": len(data), "models": rows}, fh, indent=2)
    print(f"D5-{a.stream}: {len(data)} sessions, temporal/online")
    print(f"  {'model':8} {'acc@1':>6} {'acc@5':>6} {'nll':>6} {'lat_ms':>9}")
    for r in rows:
        print(f"  {r['model']:8} {r['acc_at_1']:>6.3f} {r['acc_at_5']:>6.3f} "
              f"{r['nll_bits']:>6.2f} {r['latency_ms']:>9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
