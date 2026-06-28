"""
Behavioural re-identification probe (no LLM, fast).

Thesis (the lightweight-ToM line, made rigorous on the "model is a model of THIS
person" claim): a tiny per-user variable-order-Markov model is a behavioural
FINGERPRINT. Enrol one model per user on a slice of their stream; then take an
unseen probe segment from one user and assign it to the user whose model gives
it the highest likelihood. If re-identification accuracy is far above chance
(1/N), users are behaviourally idiosyncratic and the predictor individuates
them --- the empirical core of "a per-user model is a model of that mind".

Protocol: per user, split their sessions into ENROL (first ~60%) and PROBE
(rest). Fit a model per user on ENROL. Each PROBE session is one trial: score
its mean per-step log-prob of the realised next action under EVERY user's
model (floored), assign argmax. Report top-1 / top-3 re-id accuracy vs chance.

Usage: python3 scripts/reid_probe.py --dataset D3 --model markov1
"""
from __future__ import annotations
import sys, json, math, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.pst import PST            # noqa: E402
from markov_k import MarkovK              # noqa: E402
from data_utils import load_D1, load_D3, load_D4  # noqa: E402

FLOOR = 1e-6


class Unigram:
    """Context-free frequency profile (add-alpha). Re-id with this isolates
    'which actions the user uses' (vocabulary) from sequential structure."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.counts = {}
        self.total = 0
        self.vocab = set()

    def fit(self, sequences):
        for seq in sequences:
            for a in seq:
                self.counts[a] = self.counts.get(a, 0) + 1
                self.total += 1
                self.vocab.add(a)
        return self

    def predict(self, context):
        V = max(len(self.vocab), 1)
        denom = self.total + self.alpha * V
        return {a: (self.counts.get(a, 0) + self.alpha) / denom for a in self.vocab}


def build(name, depth):
    if name == "unigram":
        return Unigram()
    if name == "markov1":
        return MarkovK(k=1)
    if name == "markov2":
        return MarkovK(k=2)
    if name == "pst":
        return PST(max_depth=depth, min_count=2, min_gain=0.01)
    raise ValueError(name)


def logscore(model, seq, depth):
    """Mean per-step log2-prob of the realised next action under `model`."""
    if len(seq) < 2:
        return None
    tot, n = 0.0, 0
    for t in range(len(seq) - 1):
        ctx = seq[max(0, t - depth):t]
        truth = seq[t + 1]
        p = model.predict(ctx).get(truth, 0.0)
        tot += math.log2(max(p, FLOOR))
        n += 1
    return tot / n if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D3", choices=["D1", "D3", "D4"])
    ap.add_argument("--model", default="markov1", choices=["unigram", "markov1", "markov2", "pst"])
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--enrol-frac", type=float, default=0.6)
    args = ap.parse_args()

    users = {"D1": load_D1, "D3": load_D3, "D4": load_D4}[args.dataset]()
    uids = sorted(users)
    N = len(uids)

    enrol, probe = {}, {}
    for u in uids:
        sess = [s for s in users[u] if len(s) >= 3]
        if len(sess) >= 2:
            k = max(1, int(args.enrol_frac * len(sess)))
            enrol[u] = sess[:k]
            probe[u] = sess[k:] if len(sess) > k else []
        elif len(sess) == 1:                      # single long stream: split positionally
            s = sess[0]
            cut = max(3, int(args.enrol_frac * len(s)))
            enrol[u] = [s[:cut]]
            probe[u] = [s[cut:]] if len(s) - cut >= 3 else []
        else:
            enrol[u] = []; probe[u] = []

    models = {}
    for u in uids:
        if enrol[u]:
            models[u] = build(args.model, args.depth).fit(enrol[u])

    trials = correct1 = correct3 = 0
    conf = {}
    for u in uids:
        if u not in models:
            continue
        for ps in probe[u]:
            scores = []
            for v in uids:
                if v in models:
                    sc = logscore(models[v], ps, args.depth)
                    if sc is not None:
                        scores.append((sc, v))
            if not scores:
                continue
            scores.sort(reverse=True)             # higher log-prob = better
            ranked = [v for _, v in scores]
            trials += 1
            correct1 += int(ranked[0] == u)
            correct3 += int(u in ranked[:3])
            conf[(u, ranked[0])] = conf.get((u, ranked[0]), 0) + 1

    chance = 1.0 / max(len(models), 1)
    summary = {
        "dataset": args.dataset, "model": args.model, "users": len(models),
        "probe_trials": trials,
        "reid_acc_top1": round(correct1 / max(trials, 1), 4),
        "reid_acc_top3": round(correct3 / max(trials, 1), 4),
        "chance_top1": round(chance, 4),
        "lift_over_chance": round((correct1 / max(trials, 1)) / chance, 2) if trials else None,
    }
    outp = ROOT / "results" / f"reid_probe_{args.dataset}_{args.model}.json"
    outp.write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
