"""D4 builder — WebLINX action-type sequences in D1 on-disk format.

Reads the cached parquet files under data/D4-weblinx/_raw/ (produced
by scripts/explore_d4_weblinx.py), groups by `demo`, extracts the
action-function-name tokens (alphabet of 7: click, say, text_input,
scroll, load, submit, change), and materialises the top-12
demonstrations by event count to data/D4-weblinx/USER0.data ...
USER11.data in the D1 on-disk format (`**SOF**\\n<tok>\\n...\\n**EOF**\\n`).

Cohort selection rule (mirrors D3): top-N by event count across the
train+val combined set, N=12. Provenance written to
data/D4-weblinx/cohort_selection.txt.

Framing-shift acknowledgement (per spec + SOURCE.md): WebLINX is a
human-web-navigation corpus; the "user" unit here is one stable
DEMONSTRATION (no per-annotator id is exposed in the dataset).
Leave-one-user-out at evaluation time is therefore
leave-one-demonstration-out. This is documented in §1+§4 of the paper.

License: derived USER<k>.data files inherit WebLINX's
CC BY-NC-SA 4.0 — see data/D4-weblinx/_raw/SOURCE.md license fence.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "D4-weblinx" / "_raw"
OUT = ROOT / "data" / "D4-weblinx"

COHORT_SIZE = 12

ACTION_TYPE_RE = re.compile(r"^\s*(\w+)\s*\(")


def action_type(action: str) -> str:
    if not isinstance(action, str):
        return "UNK"
    m = ACTION_TYPE_RE.match(action)
    return m.group(1) if m else "UNK"


def load_combined() -> pd.DataFrame:
    frames = []
    for split in ("train", "validation"):
        p = RAW / f"chat_{split}.parquet"
        if not p.exists():
            raise RuntimeError(
                f"missing {p}; run scripts/explore_d4_weblinx.py first")
        df = pd.read_parquet(p, columns=["demo", "turn", "action"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["atype"] = df["action"].map(action_type)
    # Drop unparseable rows (defensive; explore showed 0 UNK on the 7-vocab).
    n_unk = (df["atype"] == "UNK").sum()
    if n_unk:
        print(f"[warn] dropping {n_unk} UNK rows")
        df = df[df["atype"] != "UNK"]
    return df


def select_cohort(df: pd.DataFrame, n: int = COHORT_SIZE) -> list[str]:
    counts = df.groupby("demo").size().sort_values(ascending=False)
    return counts.head(n).index.tolist()


def write_user_file(df_demo: pd.DataFrame, path: Path) -> int:
    """Write one demo as a USER<k>.data file in D1 format. Returns events."""
    seq = df_demo.sort_values("turn")["atype"].tolist()
    with path.open("w") as f:
        f.write("**SOF**\n")
        for tok in seq:
            f.write(f"{tok}\n")
        f.write("**EOF**\n")
    return len(seq)


def main():
    df = load_combined()
    cohort = select_cohort(df)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, demo_id in enumerate(cohort):
        demo_df = df[df["demo"] == demo_id]
        n = write_user_file(demo_df, OUT / f"USER{k}.data")
        rows.append((k, demo_id, n))
        print(f"USER{k}.data  demo={demo_id}  events={n}")
    # Provenance.
    prov = OUT / "cohort_selection.txt"
    with prov.open("w") as f:
        f.write("# D4 WebLINX cohort — top-12 demos by event count "
                "(train+val combined)\n")
        f.write("# Format: USER<k>  demo_id  event_count\n")
        for k, demo_id, n in rows:
            f.write(f"USER{k}\t{demo_id}\t{n}\n")
    print(f"[write] {prov}")
    total = sum(r[2] for r in rows)
    print(f"[total] {len(rows)} users, {total} events, "
          f"min={min(r[2] for r in rows)} max={max(r[2] for r in rows)}")


if __name__ == "__main__":
    main()
