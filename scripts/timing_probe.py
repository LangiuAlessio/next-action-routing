"""
Timing ("when") probe on D3 (no LLM, fast). Original-promise axis: predict not
only WHAT the next action is but WHEN it occurs.

Question: is the inter-arrival gap to the next action predictable from the
behavioural context, beyond the user's marginal gap distribution? We compare,
per user (enrol/probe split on sessions), three predictors of log10(dt+1):
  - marginal      : user's global median log-gap
  - by-action     : median log-gap conditioned on the CURRENT action
  - by-bigram     : median log-gap conditioned on (prev, current) actions
Metric on probe: median absolute error (MAE, log10 units) and Spearman rank
correlation between predicted and actual log-gap. If conditioning cuts MAE
substantially, timing is action-predictable: the model anticipates "when".

Usage: python3 scripts/timing_probe.py [--gz <path>] [--idle-gap-s 1800]
"""
from __future__ import annotations
import sys, json, math, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_d3_raw import parse                # noqa: E402
from build_d3_eclipse import normalise_cmd_id    # noqa: E402

GZ = ROOT / "data" / "D3-eclipse" / "_raw" / "filtered_usagedata_200912.csv.gz"


def sessionise_timed(events, idle_gap_s):
    """events [(ts_s, cmd)] sorted -> list of sessions, each [(cmd, dt_to_next or None)]."""
    sessions, cur, last = [], [], None
    for ts, cmd in events:
        if last is not None and ts - last > idle_gap_s and cur:
            sessions.append(cur); cur = []
        cur.append([cmd, ts]); last = ts
    if cur:
        sessions.append(cur)
    out = []
    for s in sessions:
        seq = []
        for i in range(len(s) - 1):
            dt = s[i + 1][1] - s[i][1]            # seconds to next action
            seq.append((s[i][0], dt))
        if len(seq) >= 3:
            out.append(seq)
    return out


def spearman(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2:      # constant predictor -> undefined
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    vy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return cov / (vx * vy) if vx and vy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default=str(GZ))
    ap.add_argument("--events-json", default=None, help="pre-parsed {user:[[ts,action],...]} (e.g. E1 retail); skips UDC parse+normalise")
    ap.add_argument("--tag", default=None, help="output tag override")
    ap.add_argument("--idle-gap-s", type=int, default=1800)
    ap.add_argument("--cohort", type=int, default=12)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N ranked users (cross-cohort replication)")
    ap.add_argument("--enrol-frac", type=float, default=0.6)
    args = ap.parse_args()

    if args.events_json:                          # cross-domain corpus (action token already final)
        ev = json.loads(Path(args.events_json).read_text())
        raw = {u: [(int(t), a) for t, a in seq] for u, seq in ev.items()}
        normalise = lambda c: c                   # noqa: E731
    else:
        raw = parse(Path(args.gz))                # {user: [(ts_s, cmd_id)]}
        normalise = normalise_cmd_id
    ranked = sorted(raw.items(), key=lambda kv: len(kv[1]), reverse=True)
    ranked = [(u, e) for u, e in ranked if len(e) >= 50][args.skip: args.skip + args.cohort]

    per_user = []
    all_gaps = []
    for uid, evs in ranked:
        norm = sorted([(ts, normalise(c)) for ts, c in evs if normalise(c)],
                      key=lambda x: x[0])
        sess = sessionise_timed(norm, args.idle_gap_s)
        if len(sess) >= 2:
            per_user.append((uid, sess))
            for s in sess:
                all_gaps += [dt for _, dt in s]

    def lg(dt):
        return math.log10(dt + 1.0)

    res = {"marginal": [], "by_action": [], "by_bigram": []}
    sp = {"marginal": [], "by_action": [], "by_bigram": []}
    nprobe = 0
    for uid, sess in per_user:
        k = max(1, int(args.enrol_frac * len(sess)))
        enrol, probe = sess[:k], sess[k:]
        if not probe:
            continue
        # fit conditional median tables on enrol
        g_all, by_a, by_b = [], {}, {}
        for s in enrol:
            prev = None
            for cmd, dt in s:
                g_all.append(lg(dt))
                by_a.setdefault(cmd, []).append(lg(dt))
                by_b.setdefault((prev, cmd), []).append(lg(dt))
                prev = cmd
        if not g_all:
            continue
        med_all = statistics.median(g_all)
        med_a = {a: statistics.median(v) for a, v in by_a.items()}
        med_b = {b: statistics.median(v) for b, v in by_b.items()}
        # evaluate on probe
        err = {"marginal": [], "by_action": [], "by_bigram": []}
        act, pa, pb, pm = [], [], [], []
        for s in probe:
            prev = None
            for cmd, dt in s:
                y = lg(dt)
                p_m = med_all
                p_a = med_a.get(cmd, med_all)
                p_b = med_b.get((prev, cmd), med_a.get(cmd, med_all))
                err["marginal"].append(abs(y - p_m))
                err["by_action"].append(abs(y - p_a))
                err["by_bigram"].append(abs(y - p_b))
                act.append(y); pm.append(p_m); pa.append(p_a); pb.append(p_b)
                prev = cmd
        if not act:
            continue
        nprobe += len(act)
        for kkey, preds in (("marginal", pm), ("by_action", pa), ("by_bigram", pb)):
            res[kkey].append(statistics.median(err[kkey]))
            sp[kkey].append(spearman(preds, act))

    def mean(v):
        return round(sum(v) / len(v), 4) if v else None

    # per-user robustness: in how many users does conditioning beat marginal?
    users_better = sum(1 for i in range(len(res["marginal"]))
                       if res["by_bigram"][i] < res["marginal"][i])
    users_better_pos_corr = sum(1 for c in sp["by_bigram"] if c > 0.1)

    gaps_sorted = sorted(all_gaps)
    summary = {
        "users": len(res["marginal"]), "probe_steps": nprobe,
        "gap_seconds": {
            "median": gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else None,
            "p10": gaps_sorted[int(0.1 * len(gaps_sorted))] if gaps_sorted else None,
            "p90": gaps_sorted[int(0.9 * len(gaps_sorted))] if gaps_sorted else None,
        },
        "MAE_log10gap_(lower_better)": {k: mean(v) for k, v in res.items()},
        "spearman_pred_vs_actual": {k: mean(v) for k, v in sp.items()},
        "per_user_robustness": {
            "users_bigram_beats_marginal": f"{users_better}/{len(res['marginal'])}",
            "users_spearman_gt_0.1": f"{users_better_pos_corr}/{len(sp['by_bigram'])}",
        },
    }
    m = summary["MAE_log10gap_(lower_better)"]
    if m["marginal"] and m["by_bigram"]:
        summary["MAE_reduction_bigram_vs_marginal_pct"] = round(
            100 * (m["marginal"] - m["by_bigram"]) / m["marginal"], 1)
    tag = args.tag or ("D3" if args.skip == 0 else f"D3_skip{args.skip}")
    outp = ROOT / "results" / f"timing_probe_{tag}.json"
    outp.write_text(json.dumps({"summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
