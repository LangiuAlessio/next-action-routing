"""
Calibrated 'when-head' evaluation (no LLM). Hardens the timing result (#2) into
a proper predictive task: predict the DISTRIBUTION of the next inter-arrival gap
(quantiles q10/q50/q90 of log10(gap+1)) conditioned on the action context, and
score it with a proper rule + calibration + sharpness.

Models (per user, enrol->probe), each a conditional empirical quantile predictor
with backoff to the coarser model when a cell is sparse:
  marginal   : user's global gap quantiles
  by_action  : quantiles conditioned on current action
  by_bigram  : quantiles conditioned on (prev, current) action

Metrics on probe (log10 space):
  pinball  : mean pinball/quantile loss over tau in {.1,.5,.9} (proper, lower=better)
  coverage : P(actual <= q_tau) should ~= tau (calibration)
  sharpness: mean (q90 - q10) interval width (lower=sharper, only meaningful if calibrated)

Usage:
  python3 scripts/when_head_eval.py --dataset D3 --idle-gap-s 1800
  python3 scripts/when_head_eval.py --events-json data/W1-weblog/events.json --tag W1 --idle-gap-s 1800
"""
from __future__ import annotations
import sys, json, math, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_d3_raw import parse                # noqa: E402
from build_d3_eclipse import normalise_cmd_id    # noqa: E402
from timing_probe import sessionise_timed        # noqa: E402

GZ = ROOT / "data" / "D3-eclipse" / "_raw" / "filtered_usagedata_200912.csv.gz"
TAUS = [0.1, 0.5, 0.9]
MIN_CELL = 8


def lg(dt):
    return math.log10(dt + 1.0)


def quantiles(vals, taus):
    s = sorted(vals)
    return {t: s[int(t * (len(s) - 1))] for t in taus} if s else None


def pinball(y, q, tau):
    d = y - q
    return tau * d if d >= 0 else (tau - 1) * d


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D3")
    ap.add_argument("--events-json", default=None)
    ap.add_argument("--gz", default=str(GZ))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--idle-gap-s", type=int, default=1800)
    ap.add_argument("--enrol-frac", type=float, default=0.6)
    ap.add_argument("--cohort", type=int, default=24)
    args = ap.parse_args()

    users = load(args)
    uids = sorted(users, key=lambda u: -len(users[u]))[:args.cohort]

    # eta^2: fraction of log-gap variance explained by the action / bigram
    eta_groups_a, eta_groups_b, eta_all = {}, {}, []

    agg = {m: {"pinball": [], "sharp": [], "cov": {t: [] for t in TAUS}}
           for m in ("marginal", "by_action", "by_bigram")}
    sharper_users = 0
    pinball_better_users = 0
    n_users = 0
    for u in uids:
        # sessions of [(action, dt_next)]
        norm_seq = [(t, a) for t, a in users[u]]
        sess = sessionise_timed(norm_seq, args.idle_gap_s)
        if len(sess) < 2:
            continue
        k = max(1, int(args.enrol_frac * len(sess)))
        enrol, probe = sess[:k], sess[k:]
        if not probe:
            continue
        g_all, by_a, by_b = [], {}, {}
        for s in enrol:
            prev = None
            for a, dt in s:
                g_all.append(lg(dt)); by_a.setdefault(a, []).append(lg(dt))
                by_b.setdefault((prev, a), []).append(lg(dt))
                eta_all.append(lg(dt))
                eta_groups_a.setdefault(a, []).append(lg(dt))
                eta_groups_b.setdefault((prev, a), []).append(lg(dt))
                prev = a
        if len(g_all) < 10:
            continue
        qm = quantiles(g_all, TAUS)
        qa = {a: quantiles(v, TAUS) for a, v in by_a.items() if len(v) >= MIN_CELL}
        qb = {b: quantiles(v, TAUS) for b, v in by_b.items() if len(v) >= MIN_CELL}

        per = {m: {"pin": [], "width": [], "cov": {t: [] for t in TAUS}}
               for m in agg}
        for s in probe:
            prev = None
            for a, dt in s:
                y = lg(dt)
                preds = {
                    "marginal": qm,
                    "by_action": qa.get(a, qm),
                    "by_bigram": qb.get((prev, a), qa.get(a, qm)),
                }
                for m, q in preds.items():
                    per[m]["pin"].append(sum(pinball(y, q[t], t) for t in TAUS) / len(TAUS))
                    per[m]["width"].append(q[0.9] - q[0.1])
                    for t in TAUS:
                        per[m]["cov"][t].append(1.0 if y <= q[t] else 0.0)
                prev = a
        if not per["marginal"]["pin"]:
            continue
        n_users += 1

        def mean(v):
            return sum(v) / len(v)
        for m in agg:
            agg[m]["pinball"].append(mean(per[m]["pin"]))
            agg[m]["sharp"].append(mean(per[m]["width"]))
            for t in TAUS:
                agg[m]["cov"][t].append(mean(per[m]["cov"][t]))
        if mean(per["by_bigram"]["width"]) < mean(per["marginal"]["width"]):
            sharper_users += 1
        if mean(per["by_bigram"]["pin"]) < mean(per["marginal"]["pin"]):
            pinball_better_users += 1

    def m(v):
        return round(sum(v) / len(v), 4) if v else None

    def eta2(groups, allv, min_g=8):
        if len(allv) < 2:
            return None
        gm = sum(allv) / len(allv)
        sst = sum((x - gm) ** 2 for x in allv)
        ssb = sum(len(v) * (sum(v) / len(v) - gm) ** 2
                  for v in groups.values() if len(v) >= min_g)
        return round(ssb / sst, 4) if sst > 0 else None

    out = {"corpus": args.tag or args.dataset, "users": n_users,
           "eta2_action_to_gap": eta2(eta_groups_a, eta_all),
           "eta2_bigram_to_gap": eta2(eta_groups_b, eta_all),
           "pinball_loss_(lower_better)": {k: m(agg[k]["pinball"]) for k in agg},
           "interval_width_q10_q90_(sharpness)": {k: m(agg[k]["sharp"]) for k in agg},
           "coverage_(target=tau)": {
               k: {f"q{int(t*100)}": m(agg[k]["cov"][t]) for t in TAUS} for k in agg},
           "robustness": {
               "users_bigram_sharper_than_marginal": f"{sharper_users}/{n_users}",
               "users_bigram_pinball_better": f"{pinball_better_users}/{n_users}"}}
    if out["pinball_loss_(lower_better)"]["marginal"] and out["pinball_loss_(lower_better)"]["by_bigram"]:
        b = out["pinball_loss_(lower_better)"]
        out["pinball_reduction_bigram_vs_marginal_pct"] = round(
            100 * (b["marginal"] - b["by_bigram"]) / b["marginal"], 1)
    tag = args.tag or args.dataset
    (ROOT / "results" / f"when_head_{tag}.json").write_text(json.dumps({"summary": out}, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
