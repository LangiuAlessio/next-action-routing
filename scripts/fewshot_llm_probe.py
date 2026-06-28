"""
Pivotal probe: does a FAIR few-shot LLM capture the oracle headroom on the
HARD (high-PST-entropy) steps that classical per-user models miss?

Fairness: the LLM is given THIS user's own recent action log in-context
(the same information the online classical model has accumulated), plus the
current context, and asked for the next action. If it cannot beat a tiny
Markov-1 / PST on the hard steps, then "counting beats reasoning" for personal
next-action prediction is established even against a fair few-shot LLM --- a
strong result for the PST-as-lightweight-ToM line. If it does beat them, the
entropy-gated hybrid is viable --- also a strong (positive) result.

Cost-capped: LLM is called ONLY on gated hard steps, up to MAX_CALLS total.

Usage:
  python3 scripts/fewshot_llm_probe.py --dataset D3 --model claude-haiku-4-5-20251001 \
      --users 4 --max-calls 80 --hard-quantile 0.80 --window 120
"""
from __future__ import annotations
import sys, json, math, argparse, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collections import Counter           # noqa: E402
from baselines.pst import PST            # noqa: E402
from markov_k import MarkovK              # noqa: E402
from data_utils import load_D1, load_D3, load_D4, leave_one_user_out  # noqa: E402


def shannon_entropy(dist: dict) -> float:
    tot = sum(dist.values()) or 1.0
    h = 0.0
    for v in dist.values():
        p = v / tot
        if p > 0:
            h -= p * math.log2(p)
    return h


def top1(dist: dict):
    return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None


SYS = ("You predict the next action of ONE specific user from their own action "
       "log. Actions are opaque tokens. Output ONLY a JSON object, nothing else.")

PROMPT = """\
This single user's recent action log (oldest to newest), space-separated tokens:
{hist}

Current context (the user's most recent actions): {ctx}

Allowed next-action tokens (choose exactly one): {vocab}

Predict the SINGLE most likely next token for THIS user given their habits.
Output exactly: {{"next": "<one token from the allowed list>"}}"""


def llm_predict(history, ctx, vocab, model, endpoint, timeout=90):
    """Clean completion via Ollama /api/generate (format=json, temp 0)."""
    prompt = SYS + "\n\n" + PROMPT.format(
        hist=" ".join(history), ctx=" ".join(ctx), vocab=" ".join(sorted(vocab)))
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 64},
    }).encode()
    req = urllib.request.Request(f"{endpoint}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
    except Exception:
        return None
    text = (resp.get("response") or "").strip()
    import re
    m = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
    if not m:
        return None
    try:
        val = json.loads(m[-1].group()).get("next")
        return val.strip() if isinstance(val, str) else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D3", choices=["D1", "D3", "D4"])
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--endpoint", default="http://192.168.1.72:11434")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--hard-quantile", type=float, default=0.80)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    loader = {"D1": load_D1, "D3": load_D3, "D4": load_D4}[args.dataset]
    users = loader()
    splits = leave_one_user_out(users)[: args.users]

    rows = []   # per hard step: (truth, pst_top1, m1_top1, llm_top1, entropy)
    calls = 0
    for test_uid, train, test in splits:
        if calls >= args.max_calls:
            break
        pst = PST(max_depth=args.depth, min_count=2, min_gain=0.01).fit(train)
        m1 = MarkovK(k=1).fit(train)
        vocab = sorted({a for seq in test for a in seq})  # this user's own repertoire (bounded, fair candidate set)
        # online pass: PST entropy per step; record the user's revealed log
        # stream tuple: (ctx, truth, revealed, pst_top1, m1_top1, uni_top1, entropy)
        stream = []
        revealed = []
        pst_e = PST(max_depth=args.depth, min_count=2, min_gain=0.01).fit(train)
        m1_e = MarkovK(k=1).fit(train)
        uni = Counter(a for s in train for a in s)   # personal base-rate (train + revealed)
        ents = []
        for seq in test:
            if len(seq) < 3:
                continue
            for t in range(len(seq) - 1):
                ctx = seq[max(0, t - args.depth):t]
                truth = seq[t + 1]
                dist = pst_e.predict(ctx)
                e = shannon_entropy(dist)
                ents.append(e)
                uni_top1 = uni.most_common(1)[0][0] if uni else None
                stream.append((list(ctx), truth, list(revealed), top1(dist),
                               top1(m1_e.predict(ctx)), uni_top1, e))
                pst_e.update(ctx, truth); m1_e.update(ctx, truth)
                revealed.append(seq[t]); uni[seq[t]] += 1
            revealed.append(seq[-1]); uni[seq[-1]] += 1
        if not ents:
            continue
        ents_sorted = sorted(ents)
        thr = ents_sorted[int(args.hard_quantile * (len(ents_sorted) - 1))]
        hard = [s for s in stream if s[6] >= thr]
        for ctx, truth, pfx, pst_t, m1_t, uni_t, e in hard:
            if calls >= args.max_calls:
                break
            window = (pfx + ctx)[-args.window:]
            llm_t = llm_predict(window, ctx, vocab, args.model, args.endpoint)
            calls += 1
            rows.append({"uid": test_uid, "truth": truth, "pst": pst_t,
                         "m1": m1_t, "uni": uni_t, "llm": llm_t, "entropy": round(e, 3)})
            print(f"  [{calls}/{args.max_calls}] u={test_uid} H={e:.2f} "
                  f"truth={truth} pst={pst_t} m1={m1_t} uni={uni_t} llm={llm_t}", flush=True)

    n = len(rows)
    def acc(key):
        return sum(1 for r in rows if r[key] == r["truth"]) / max(n, 1)
    # realistic hybrid on hard steps: use LLM where it answered, else PST
    hyb = sum(1 for r in rows if (r["llm"] if r["llm"] else r["pst"]) == r["truth"]) / max(n, 1)
    summary = {
        "dataset": args.dataset, "model": args.model, "hard_steps_scored": n,
        "hard_quantile": args.hard_quantile,
        "acc1_on_hard": {"pst": round(acc("pst"), 4), "m1": round(acc("m1"), 4),
                          "uni": round(acc("uni"), 4), "llm": round(acc("llm"), 4),
                          "hybrid_llm_on_hard": round(hyb, 4)},
        "llm_valid_answers": sum(1 for r in rows if r["llm"] is not None),
    }
    outp = ROOT / "results" / f"fewshot_llm_probe_{args.dataset}.json"
    outp.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
