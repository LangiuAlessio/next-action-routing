"""Raw Eclipse-UDC filtered dump -> data/D3-eclipse/USER<k>.data in the
D1 on-disk format (**SOF** / one token per line / **EOF**).

Cohort = top-N users by command-event count (>= min events).
Sessionise by idle gap (default 30 min). Deterministic.

parse() (validate_d3_raw) already: filters kind=="command", converts
ms->int seconds, returns {user: [(ts_seconds:int, command_id:str), ...]}.
So this builder only normalises the command id, sorts, sessionises, writes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_d3_raw import parse  # single point of raw-format coupling

OUT = ROOT / "data" / "D3-eclipse"


def normalise_cmd_id(cid: str):
    """Eclipse command id -> compact stable token (last <=3 dotted
    segments, lower-case). Keeps Sigma large but readable; deterministic."""
    cid = cid.strip().lower()
    if not cid:
        return None
    parts = [p for p in cid.split(".") if p]
    if not parts:
        return None
    tok = ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts)
    return tok if tok and tok[0].isalnum() else None


def sessionise(events, idle_gap_s: int = 1800):
    """events: [(ts_seconds, cmd_id)] sorted by ts -> [[cmd,...], ...].
    A gap > idle_gap_s starts a new session."""
    sessions, cur, last = [], [], None
    for ts, cmd in events:
        if last is not None and ts - last > idle_gap_s and cur:
            sessions.append(cur)
            cur = []
        cur.append(cmd)
        last = ts
    if cur:
        sessions.append(cur)
    return sessions


def build(raw_path: Path, cohort: int = 12, min_events: int = 50,
          idle_gap_s: int = 1800, min_seq_len: int = 3):
    raw = parse(raw_path)  # {user: [(ts_s:int, cmd_id:str)]}
    ranked = sorted(raw.items(), key=lambda kv: len(kv[1]), reverse=True)
    ranked = [(u, evs) for u, evs in ranked if len(evs) >= min_events][:cohort]
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, (uid, evs) in enumerate(ranked):
        norm = []
        for ts, cid in evs:
            c = normalise_cmd_id(cid)
            if c:
                norm.append((ts, c))
        norm.sort(key=lambda x: x[0])
        sessions = [s for s in sessionise(norm, idle_gap_s)
                    if len(s) >= min_seq_len]
        if not sessions:
            continue
        lines = []
        for s in sessions:
            lines.append("**SOF**")
            lines += s
            lines.append("**EOF**")
        (OUT / f"USER{i}.data").write_text("\n".join(lines) + "\n")
        written += 1
    print(f"D3: wrote {written} users to {OUT} "
          f"(cohort={cohort}, min_events={min_events}, idle_gap_s={idle_gap_s})")
    return written


if __name__ == "__main__":
    n = build(Path(sys.argv[1]))
    sys.exit(0 if n >= 5 else 2)
