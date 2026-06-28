"""
What+When probe (no LLM, fast): does TIMING improve prediction of WHAT comes next?

Causal feature (no leak): at the prediction point (just did action a_cur), the
inter-arrival gap that LED INTO a_cur is known; the gap to the next action is
not. We test whether conditioning next-action prediction on the bucketed recent
gap beats a plain Markov-1 (last-action) model.

  base : P(next | a_cur)                      [Markov-1]
  aug  : P(next | a_cur, gap_bucket(prev_gap))  backing off to base when the
         (a_cur, bucket) cell is sparse (so aug >= base unless timing adds info)

Per user (enrol/probe positional split), report acc@1 and NLL(bits) for both,
paired across users. Buckets = per-user quantiles of prev_gap from enrol.

Usage:
  python3 scripts/whatwhen_probe.py --dataset D3
  python3 scripts/whatwhen_probe.py --events-json data/W1-weblog/events.json --tag W1 --idle-gap-s 1800
"""
from __future__ import annotations
import sys, json, math, argparse, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_d3_raw import parse                # noqa: E402
from build_d3_eclipse import normalise_cmd_id    # noqa: E402

GZ = ROOT / "data" / "D3-eclipse" / "_raw" / "filtered_usagedata_200912.csv.gz"
ALPHA = 0.3
MIN_CELL = 5          # min counts for the (a_cur,bucket) cell to be used over backoff
NBUCKET = 4


def load(args):
    if args.events_json:
        ev = json.loads(Path(args.events_json).read_text())
        raw = {u: [(int(t), a) for t, a in seq] for u, seq in ev.items()}
        norm = lambda c: c                        # noqa: E731
    else:
        raw = parse(Path(args.gz))
        norm = normalise_cmd_id
    out = {}
    for u, evs in raw.items():
        seq = sorted([(t, norm(c)) for t, c in evs if norm(c)], key=lambda x: x[0])
        if len(seq) >= 50:
            out[u] = seq
    return out


def quantile_buckets(gaps, nb):
    g = sorted(gaps)
    if not g:
        return []
    return [g[int(q * (len(g) - 1))] for q in [i / nb for i in range(1, nb)]]


def bucket_of(gap, edges):
    b = 0
    for e in edges:
        if gap > e:
            b += 1
        else:
            break
    return b


def predict_dist(counts, key, vocab):
    row = counts.get(key, {})
    tot = sum(row.values()) + ALPHA * len(vocab)
    return {a: (row.get(a, 0) + ALPHA) / tot for a in vocab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D3")
    ap.add_argument("--events-json", default=None)
    ap.add_argument("--gz", default=str(GZ))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--enrol-frac", type=float, default=0.6)
    ap.add_argument("--cohort", type=int, default=24)
    args = ap.parse_args()

    users = load(args)
    uids = sorted(users, key=lambda u: -len(users[u]))[:args.cohort]

    acc_base, acc_aug, nll_base, nll_aug = [], [], [], []
    for u in uids:
        seq = users[u]
        cut = max(10, int(args.enrol_frac * len(seq)))
        enrol, probe = seq[:cut], seq[cut:]
        if len(probe) < 5:
            continue
        vocab = sorted({a for _, a in seq})
        gaps = [enrol[i][0] - enrol[i - 1][0] for i in range(1, len(enrol))]
        edges = quantile_buckets(gaps, NBUCKET)

        base = defaultdict(lambda: defaultdict(int))   # a_cur -> next
        aug = defaultdict(lambda: defaultdict(int))    # (a_cur,bucket) -> next
        for i in range(1, len(enrol) - 1):
            a_cur = enrol[i][1]; nxt = enrol[i + 1][1]
            gp = enrol[i][0] - enrol[i - 1][0]
            base[a_cur][nxt] += 1
            aug[(a_cur, bucket_of(gp, edges))][nxt] += 1

        ab = aa = 0; nb = na = 0.0; m = 0
        for i in range(1, len(probe) - 1):
            a_cur = probe[i][1]; truth = probe[i + 1][1]
            gp = probe[i][0] - probe[i - 1][0]
            bkey = (a_cur, bucket_of(gp, edges))
            d_base = predict_dist(base, a_cur, vocab)
            # aug: use cell only if dense enough, else back off to base
            if sum(aug.get(bkey, {}).values()) >= MIN_CELL:
                d_aug = predict_dist(aug, bkey, vocab)
            else:
                d_aug = d_base
            top_b = max(d_base, key=d_base.get)
            top_a = max(d_aug, key=d_aug.get)
            ab += int(top_b == truth); aa += int(top_a == truth)
            nb += -math.log2(max(d_base.get(truth, 1e-12), 1e-12))
            na += -math.log2(max(d_aug.get(truth, 1e-12), 1e-12))
            m += 1
        if m == 0:
            continue
        acc_base.append(ab / m); acc_aug.append(aa / m)
        nll_base.append(nb / m); nll_aug.append(na / m)

    def mean(v):
        return round(sum(v) / len(v), 4) if v else None

    n = len(acc_base)
    acc_up = sum(1 for i in range(n) if acc_aug[i] > acc_base[i])
    nll_dn = sum(1 for i in range(n) if nll_aug[i] < nll_base[i])
    # paired sign test on acc improvement
    from math import comb
    pos = sum(1 for i in range(n) if acc_aug[i] > acc_base[i])
    neg = sum(1 for i in range(n) if acc_aug[i] < acc_base[i])
    nn = pos + neg
    p = (sum(comb(nn, k) for k in range(min(pos, neg) + 1)) / 2 ** nn * 2) if nn else 1.0
    summary = {
        "corpus": args.tag or args.dataset, "users": n,
        "acc1": {"base": mean(acc_base), "aug_timing": mean(acc_aug),
                 "delta": round((mean(acc_aug) or 0) - (mean(acc_base) or 0), 4),
                 "users_improved": f"{acc_up}/{n}", "sign_test_p": round(min(p, 1.0), 4)},
        "nll_bits": {"base": mean(nll_base), "aug_timing": mean(nll_aug),
                     "delta_lower_better": round((mean(nll_aug) or 0) - (mean(nll_base) or 0), 4),
                     "users_improved": f"{nll_dn}/{n}"},
    }
    tag = args.tag or args.dataset
    (ROOT / "results" / f"whatwhen_probe_{tag}.json").write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
