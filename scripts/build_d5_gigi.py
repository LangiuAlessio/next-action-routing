"""Extract gigi D5-event sequences to data/D5-gigi/event.jsonl (reproducible run input)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_d5_gigi import load_D5_event


def main(argv=None):
    root = (argv or sys.argv[1:] or [os.path.expanduser("~/gigi")])[0]
    d = load_D5_event(root)
    out = Path(__file__).parent.parent / "data" / "D5-gigi" / "event.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for sess, (ts, seq) in sorted(d.items(), key=lambda kv: kv[1][0]):
            fh.write(json.dumps({"session": sess, "first_ts": ts, "seq": seq}) + "\n")
    print(f"wrote {len(d)} sessions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
