"""D4 WebLINX schema exploration — Task 2.2 step 0.

Downloads the chat/train + chat/validation parquet exports of the
McGill-NLP/WebLINX dataset (small: ~26 MB total) and computes the
per-demonstration event counts and the action-type vocabulary.

Purpose: BEFORE committing to the full builder, validate the
predicate "≥5 demos with ≥50 events each". If the predicate
fails, surface the result honestly and reconsider before building
the USER<k>.data files.

Output: results/d4_weblinx_explore.json with per-demo event counts,
action vocabulary, and validation verdict.

License note: the WebLINX dataset is CC BY-NC-SA 4.0; we cache the
parquet files locally only for this analysis. See
data/D4-weblinx/_raw/SOURCE.md for the license fence applied to any
derived artefact released as part of the bench's Zenodo bundle.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "D4-weblinx" / "_raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Public parquet exports (refresh: re-check via hub_repo_details if the
# server returns 404 — Hugging Face occasionally rotates file paths).
PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/refs%2Fconvert%2Fparquet/chat/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/refs%2Fconvert%2Fparquet/chat/validation/0000.parquet",
}

# Action type extraction: take the leading identifier before the first
# parenthesis. Examples:
#   click(uid="...") -> click
#   say(speaker="navigator", utterance="...") -> say
#   text_input(text="...", uid="...") -> text_input
ACTION_TYPE_RE = re.compile(r"^\s*(\w+)\s*\(")


def action_type(action: str) -> str:
    """Extract the BrowserGym action function name, or 'UNK' if unparseable."""
    if not isinstance(action, str):
        return "UNK"
    m = ACTION_TYPE_RE.match(action)
    return m.group(1) if m else "UNK"


def fetch_parquet(split: str) -> Path:
    """Cache the chat/<split> parquet locally under _raw/.

    Returns the local path. Idempotent: skips if already present.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    local = RAW / f"chat_{split}.parquet"
    if local.exists() and local.stat().st_size > 0:
        return local
    url = PARQUET_URLS[split]
    print(f"[fetch] {split}: {url} -> {local}")
    # Use pandas directly: it can read remote parquet via fsspec/pyarrow.
    df = pd.read_parquet(url, columns=["demo", "turn", "action"])
    df.to_parquet(local, index=False)
    return local


def explore():
    frames = []
    for split in ("train", "validation"):
        p = fetch_parquet(split)
        df = pd.read_parquet(p, columns=["demo", "turn", "action"])
        df["split"] = split
        frames.append(df)
        print(f"[{split}] rows={len(df)} demos={df['demo'].nunique()}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[combined] rows={len(df)} demos={df['demo'].nunique()}")

    # Action type vocabulary.
    df["atype"] = df["action"].map(action_type)
    vocab = Counter(df["atype"].tolist())
    print(f"[vocab] |Σ|={len(vocab)} top={vocab.most_common()}")

    # Per-demo event count, ordered by turn.
    counts = df.groupby("demo").size().sort_values(ascending=False)
    print(f"[demos] top-20 by event count:")
    print(counts.head(20).to_string())
    print(f"[demos] >=50 events: {(counts >= 50).sum()}")
    print(f"[demos] >=30 events: {(counts >= 30).sum()}")
    print(f"[demos] >=20 events: {(counts >= 20).sum()}")
    print(f"[demos] >=10 events: {(counts >= 10).sum()}")

    # Validation predicate.
    big_demos = counts[counts >= 50]
    verdict = "PASS" if len(big_demos) >= 5 else "FAIL"
    print(f"\n[VERDICT] D4 predicate (>=5 demos with >=50 events): {verdict}")
    print(f"  qualifying demos: {len(big_demos)}")

    out = {
        "total_rows": int(len(df)),
        "total_demos": int(df["demo"].nunique()),
        "vocab_size": len(vocab),
        "vocab": dict(vocab.most_common()),
        "demos_ge_50": int((counts >= 50).sum()),
        "demos_ge_30": int((counts >= 30).sum()),
        "demos_ge_20": int((counts >= 20).sum()),
        "demos_ge_10": int((counts >= 10).sum()),
        "top20_demos": {k: int(v) for k, v in counts.head(20).items()},
        "verdict": verdict,
    }
    (RESULTS / "d4_weblinx_explore.json").write_text(json.dumps(out, indent=2))
    print(f"[write] results/d4_weblinx_explore.json")


if __name__ == "__main__":
    explore()
