#!/usr/bin/env python3
"""
run_benchmark.py — PST-vs-LLM benchmark runner.

Usage:
    python3 run_benchmark.py --dataset D2 --models all --out results/
    python3 run_benchmark.py --dataset D2 --models pst ppm --quick
    python3 run_benchmark.py --compare results/benchmark_D2_*.json

Output:
    results/benchmark_<dataset>_<model>.json
    results/comparison_<dataset>.json  (with --compare)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_utils import generate_D2_synthetic, leave_one_user_out, _INTENT_GROUPS, ACTION_TO_INTENT
from baselines.pst import PST
from baselines.ppm import PPM
from baselines.ctw import CTW
from baselines.ctw_faithful import CTWFaithful
from baselines.lz_pred import LZPredictor
from baselines.llm_react import LLMReAct
from eval.metrics import acc_at_k, bleu_action, latency_ms, intent_switch_latency
from eval.intent_cluster import kmeans

INTENT_MAP_STR = {k: i for i, k in enumerate(_INTENT_GROUPS.keys())}


def _build_model(name: str, depth: int):
    if name == "pst":
        return PST(max_depth=depth, min_count=2, min_gain=0.01)
    if name == "ppm":
        return PPM(max_order=depth)
    if name == "ctw":
        return CTW(depth=depth)
    if name == "ctw_faithful":
        return CTWFaithful(depth=depth)
    if name == "lz":
        return LZPredictor()
    if name == "react":
        return LLMReAct(model="claude-haiku-4-5-20251001", max_vocab=30)
    raise ValueError(f"Unknown model: {name}")


def get_intent(action: str) -> int:
    return INTENT_MAP_STR.get(ACTION_TO_INTENT.get(action, ""), 0)


def evaluate_model(
    model_name: str,
    users: dict,
    dataset_name: str,
    depth: int = 6,
    k_intent: int = 6,
    verbose: bool = True,
    max_steps: int | None = None,
    max_folds: int | None = None,
    ckpt_path: "Path | None" = None,
) -> dict:
    splits = leave_one_user_out(users)
    all_results = []

    def _pack():
        def mean(key):
            vals = [r[key] for r in all_results if r[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        return {
            "dataset": dataset_name,
            "model": model_name.upper(),
            "depth": depth,
            "n_users": len(all_results),
            "aggregate": {
                "acc_at_1": mean("acc_at_1"),
                "acc_at_5": mean("acc_at_5"),
                "iota": mean("iota"),
                "latency_mean_ms": mean("latency_mean_ms"),
                "bleu_action": mean("bleu_action"),
                "intent_switch_latency": mean("intent_switch_latency"),
            },
            "per_user": all_results,
        }

    # Resume: the result JSON IS the checkpoint. Reload completed folds
    # and skip those test users on rerun.
    done_uids = set()
    if ckpt_path is not None and ckpt_path.exists():
        try:
            prev = json.loads(ckpt_path.read_text())
            for r in prev.get("per_user", []):
                if r.get("user"):
                    all_results.append(r)
                    done_uids.add(r["user"])
            if verbose and done_uids:
                print(f"  [resume] {len(done_uids)} folds from checkpoint")
        except (json.JSONDecodeError, OSError):
            pass

    for idx, (test_uid, train_seqs, test_seqs) in enumerate(splits):
        if test_uid in done_uids:
            continue
        if max_folds is not None and len(all_results) >= max_folds:
            break
        if verbose:
            print(f"  [{idx+1}/{len(splits)}] {model_name} user={test_uid}", end="", flush=True)

        model = _build_model(model_name, depth)
        t0 = time.perf_counter()
        model.fit(train_seqs)
        train_ms = (time.perf_counter() - t0) * 1000

        # intent labels: ground-truth for D2, k-means for D1
        if dataset_name == "D2":
            def get_intent_fn(a):
                return get_intent(a)
        else:
            cluster = kmeans(train_seqs, k=k_intent)
            def get_intent_fn(a, _c=cluster):
                return _c.get(a, 0)

        acc1_vals, acc5_vals, iota_vals, lat_vals, bleu_vals = [], [], [], [], []
        all_preds, all_truths, all_intents = [], [], []
        steps_done = 0

        for seq in test_seqs:
            if len(seq) < 3:
                continue
            if max_steps is not None and steps_done >= max_steps:
                break
            seq_preds = []
            for t in range(len(seq) - 1):
                if max_steps is not None and steps_done >= max_steps:
                    break
                ctx = seq[max(0, t - depth): t]
                truth = seq[t + 1]

                top, ms = latency_ms(model.predict_top, ctx, k=5)
                lat_vals.append(ms)
                acc1_vals.append(acc_at_k(top, truth, k=1))
                acc5_vals.append(acc_at_k(top, truth, k=5))
                iota_vals.append(model.interpretability_score(ctx))
                pred = top[0][0] if top else ""
                seq_preds.append(pred)
                all_preds.append(pred)
                all_truths.append(truth)
                all_intents.append(get_intent_fn(truth))
                model.update(ctx, truth)
                steps_done += 1

            bleu_vals.append(bleu_action(seq_preds, seq[1:]))

        isl = intent_switch_latency(all_preds, all_truths, all_intents)

        result = {
            "user": test_uid,
            "train_ms": round(train_ms, 2),
            "acc_at_1": round(sum(acc1_vals) / max(len(acc1_vals), 1), 4),
            "acc_at_5": round(sum(acc5_vals) / max(len(acc5_vals), 1), 4),
            "iota": round(sum(iota_vals) / max(len(iota_vals), 1), 4),
            "latency_mean_ms": round(sum(lat_vals) / max(len(lat_vals), 1), 4),
            "bleu_action": round(sum(bleu_vals) / max(len(bleu_vals), 1), 4),
            "intent_switch_latency": round(isl, 3) if isl is not None else None,
        }
        all_results.append(result)
        if ckpt_path is not None:
            ckpt_path.write_text(json.dumps(_pack(), indent=2))
        if verbose:
            print(f"  acc@1={result['acc_at_1']:.3f}  ι={result['iota']:.3f}"
                  f"  lat={result['latency_mean_ms']:.3f}ms")

    return _pack()


def compare_results(result_files: list[Path]) -> dict:
    """Load multiple result JSONs and build a comparison table."""
    rows = []
    for f in sorted(result_files):
        data = json.loads(f.read_text())
        row = {"model": data["model"], "dataset": data["dataset"]}
        row.update(data["aggregate"])
        rows.append(row)
    return {"comparison": rows}


def print_table(comparison: dict):
    rows = comparison["comparison"]
    if not rows:
        return
    keys = ["model", "acc_at_1", "acc_at_5", "iota", "latency_mean_ms",
            "bleu_action", "intent_switch_latency"]
    header = f"{'Model':8s}  {'acc@1':6s}  {'acc@5':6s}  {'ι':6s}  {'lat(ms)':8s}  {'BLEU':6s}  {'ISL':6s}"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for row in rows:
        print(f"{row['model']:8s}  "
              f"{row['acc_at_1'] or 0:6.3f}  "
              f"{row['acc_at_5'] or 0:6.3f}  "
              f"{row['iota'] or 0:6.3f}  "
              f"{row['latency_mean_ms'] or 0:8.4f}  "
              f"{row['bleu_action'] or 0:6.4f}  "
              f"{row['intent_switch_latency'] or 0:6.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["D1", "D2", "D3", "D4"], default="D2")
    parser.add_argument("--models", nargs="+",
                        choices=["pst", "ppm", "ctw", "ctw_faithful", "lz", "react", "all"],
                        default=["all"])
    parser.add_argument("--max-steps-llm", type=int, default=50,
                        help="Max prediction steps per fold for LLM models (cost control)")
    parser.add_argument("--max-users-llm", type=int, default=None,
                        help="Cost-cap: max LOUO folds for LLM models "
                             "(e.g. 5 = 5 users x N steps, like D2). "
                             "Result JSON doubles as a resumable checkpoint.")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    parser.add_argument("--compare", nargs="*", metavar="FILE",
                        help="Compare existing result JSON files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # Compare mode
    if args.compare is not None:
        if args.compare:
            files = [Path(f) for f in args.compare]
        else:
            files = list(out_dir.glob(f"benchmark_{args.dataset}_*.json"))
        comp = compare_results(files)
        print_table(comp)
        out = out_dir / f"comparison_{args.dataset}.json"
        out.write_text(json.dumps(comp, indent=2))
        print(f"\nComparison saved to {out}")
        return

    if args.quick:
        args.users = 5
        args.sessions = 5

    models = ["pst", "ppm", "ctw", "lz"] if "all" in args.models else args.models

    print(f"Loading dataset {args.dataset}...")
    if args.dataset == "D2":
        users = generate_D2_synthetic(n_users=args.users, sessions_per_user=args.sessions)
    elif args.dataset == "D3":
        from data_utils import load_D3
        users = load_D3()
    elif args.dataset == "D4":
        from data_utils import load_D4
        users = load_D4()
    else:
        from data_utils import load_D1
        users = load_D1()
    print(f"  {len(users)} users, depth={args.depth}")

    all_files = []
    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")
        llm_models = {"react"}
        is_llm = model_name in llm_models
        max_steps = args.max_steps_llm if is_llm else None
        max_folds = args.max_users_llm if is_llm else None
        out_path = out_dir / f"benchmark_{args.dataset}_{model_name}.json"
        ckpt = out_path if is_llm else None
        if max_steps is not None:
            cap = f", max {max_folds} folds" if max_folds else ""
            print(f"  (LLM mode: max {max_steps} steps/fold{cap}; "
                  f"checkpoint+resume via {out_path.name})")
        result = evaluate_model(
            model_name, users, args.dataset, depth=args.depth, k_intent=6,
            max_steps=max_steps, max_folds=max_folds, ckpt_path=ckpt,
        )
        out_path.write_text(json.dumps(result, indent=2))
        all_files.append(out_path)
        agg = result["aggregate"]
        print(f"  AGGREGATE: acc@1={agg['acc_at_1']}  acc@5={agg['acc_at_5']}"
              f"  ι={agg['iota']}  lat={agg['latency_mean_ms']}ms"
              f"  ISL={agg['intent_switch_latency']}")

    if len(all_files) > 1:
        comp = compare_results(all_files)
        print_table(comp)
        (out_dir / f"comparison_{args.dataset}.json").write_text(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
