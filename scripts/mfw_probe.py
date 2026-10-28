"""Alternative-method probe: MFW/antidictionary-gated PST vs vanilla PST.

Reuses an external MFW extractor (mff_extract.py, not distributed with
this bundle; set MFF_EXTRACT_PATH to its location) and the bench's PST + D1/D3 loaders. Per LOUO fold:
  - vanilla PST (depth 6, online update) -> P(a | ctx);
  - MFW-gated: forbid any candidate a s.t. a suffix of (ctx + a) is a
    minimal forbidden word over the training actions; renormalise the
    PST distribution over the survivors; if exactly one survives, it is
    *forced* (DCA prediction principle). Fallback to ungated if all
    forbidden.
Zero LLM cost. Output: results/mfw_probe.json + printed summary.
"""
import sys, json, statistics as st
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from data_utils import load_D1, load_D3, leave_one_user_out
from run_benchmark import _build_model
# Load mff_extract by file path (do NOT add forbidden-factors-exp/src to
# sys.path: it contains an eval.py that shadows the bench's eval package).
import importlib.util as _ilu
_mff_path = Path(os.environ["MFF_EXTRACT_PATH"])  # external, not distributed
_spec = _ilu.spec_from_file_location("mff_extract", _mff_path)
_mff = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mff)
extract_mff_trie = _mff.extract_mff_trie

DEPTH = 6
MFW_MIN, MFW_MAX = 2, 6
SEP = -1


def fold_eval(train_seqs, test_seqs):
    vocab = sorted({a for s in train_seqs for a in s})
    aid = {a: i for i, a in enumerate(vocab)}
    toks = []
    for s in train_seqs:
        toks.extend(aid[a] for a in s)
        toks.append(SEP)
    mfw = extract_mff_trie(toks, min_len=MFW_MIN, max_len=MFW_MAX)
    m = _build_model("pst", DEPTH)
    m.fit(train_seqs)
    v_ok = g_ok = n = forced = forced_ok = 0
    for seq in test_seqs:
        if len(seq) < 3:
            continue
        for t in range(len(seq) - 1):
            ctx = seq[max(0, t - DEPTH):t]
            truth = seq[t + 1]
            dist = m.predict(ctx)
            if dist:
                v_top = max(dist, key=dist.get)
                v_ok += (v_top == truth)
                cid = [aid[a] for a in ctx if a in aid]
                allowed = {}
                for a, p in dist.items():
                    ia = aid.get(a)
                    forbidden = False
                    if ia is not None:
                        for L in range(MFW_MIN, MFW_MAX + 1):
                            if len(cid) >= L - 1:
                                w = tuple(cid[-(L - 1):] + [ia]) if L > 1 else (ia,)
                                if w in mfw:
                                    forbidden = True
                                    break
                    if not forbidden:
                        allowed[a] = p
                use = allowed if allowed else dist
                if allowed and len(allowed) == 1:
                    forced += 1
                    forced_ok += (next(iter(allowed)) == truth)
                g_top = max(use, key=use.get)
                g_ok += (g_top == truth)
                n += 1
            m.update(ctx, truth)
    return dict(n=n, v=v_ok / n if n else 0, g=g_ok / n if n else 0,
                forced=forced, forced_acc=(forced_ok / forced if forced else None))


def main():
    out = {}
    for name, ld in [("D1", load_D1), ("D3", load_D3)]:
        rows = [fold_eval(tr, te) for _u, tr, te in leave_one_user_out(ld())]
        v = st.mean(r["v"] for r in rows)
        g = st.mean(r["g"] for r in rows)
        wins = sum(1 for r in rows if r["g"] > r["v"])
        ties = sum(1 for r in rows if abs(r["g"] - r["v"]) < 1e-12)
        fa = [r["forced_acc"] for r in rows if r["forced_acc"] is not None]
        out[name] = dict(folds=len(rows), pst=round(v, 4), mfw_gated=round(g, 4),
                         delta=round(g - v, 4),
                         gated_better_folds=f"{wins}/{len(rows)}",
                         tie_folds=ties,
                         forced_steps=sum(r["forced"] for r in rows),
                         forced_acc=round(st.mean(fa), 4) if fa else None)
        print(f"{name}: PST={v:.4f}  MFW-gated={g:.4f}  Δ={g-v:+.4f}  "
              f"gated>PST in {wins}/{len(rows)} folds (ties {ties})  "
              f"forced={sum(r['forced'] for r in rows)} "
              f"forced_acc={out[name]['forced_acc']}")
    (ROOT / "results" / "mfw_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
