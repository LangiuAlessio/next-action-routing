"""Paired LOUO statistics from already-collected per-user results.
Bootstrap 95% CI (10k) + Wilcoxon signed-rank, Holm-corrected per dataset.
Usage: python3 scripts/stats_paired.py
"""
import json, random, math, statistics as st
from pathlib import Path

R = Path(__file__).parent.parent / "results"


def per(ds, m, k="acc_at_1"):
    return {r["user"]: r[k] for r in json.load(open(R / f"benchmark_{ds}_{m}.json"))["per_user"]}


def boot_ci(d, n=10000):
    random.seed(0)
    ms = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n)]


def wilcoxon_p(d):
    d = [x for x in d if x]
    n = len(d)
    if n < 1:
        return 1.0
    a = sorted(range(n), key=lambda i: abs(d[i]))
    rk = [0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(d[a[j]]) == abs(d[a[i]]):
            j += 1
        v = sum(range(i + 1, j + 1)) / (j - i)
        for t in range(i, j):
            rk[a[t]] = v
        i = j
    Wp = sum(rk[i] for i in range(n) if d[i] > 0)
    Wn = sum(rk[i] for i in range(n) if d[i] < 0)
    W = min(Wp, Wn)
    mu = n * (n + 1) / 4
    sig = (n * (n + 1) * (2 * n + 1) / 24) ** .5
    z = (W - mu) / sig if sig else 0
    return min(1.0, 2 * 0.5 * math.erfc(abs(z) / 2 ** .5))


def main():
    for ds, others in [("D1", ["ppm", "lz"]), ("D3", ["ppm", "ctw", "lz"])]:
        pst = per(ds, "pst")
        rows = []
        for m in others:
            o = per(ds, m)
            us = sorted(set(pst) & set(o))
            diff = [pst[u] - o[u] for u in us]
            rows.append((m, st.mean(diff), boot_ci(diff), wilcoxon_p(diff), len(us)))
        rows.sort(key=lambda r: r[3])
        k = len(rows)
        for idx, (m, md, ci, p, n) in enumerate(rows):
            holm = min(1.0, p * (k - idx))
            sig = "sig" if holm < 0.05 else "n.s."
            print(f"{ds} PST vs {m.upper():4} n={n} dMean={md:+.4f} "
                  f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] p={p:.4f} Holm={holm:.4f} {sig}")


if __name__ == "__main__":
    main()
