"""Build W1 (NASA-HTTP web-server log) per-host clickstream with per-event
timestamps, for cross-domain timing replication.

Each successful GET is one action; action token = coarsened URL (first two path
segments + extension class, analogous to D3's last-3-dotted-segments), user =
requesting host, ts = request time (epoch seconds). Writes
data/W1-weblog/events.json: {host: [[ts_seconds, token], ...]}.
"""
import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
GZ = ROOT / "data" / "W1-weblog" / "_raw" / "nasa_jul95.gz"
OUT = ROOT / "data" / "W1-weblog" / "events.json"
if len(sys.argv) >= 3:                            # build_w1_weblog.py <gz> <out.json>
    GZ = Path(sys.argv[1]); OUT = Path(sys.argv[2])

LINE = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)[^"]*" (\d{3}) (\S+)')


def coarsen(path: str) -> str:
    path = path.split("?")[0].strip().lower()
    m = re.search(r'\.([a-z0-9]{1,5})$', path)
    ext = m.group(1) if m else ("dir" if path.endswith("/") else "none")
    segs = [s for s in path.split("/") if s]
    base = "/".join(segs[:2]) if segs else "root"
    return f"{base}.{ext}"


def main(min_events=30):
    events = {}
    bad = 0
    with gzip.open(GZ, "rt", errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                bad += 1
                continue
            host, tstr, method, path, status, _ = m.groups()
            if method != "GET" or status != "200":
                continue
            try:
                ts = int(datetime.strptime(tstr, "%d/%b/%Y:%H:%M:%S %z").timestamp())
            except ValueError:
                bad += 1
                continue
            events.setdefault(host, []).append([ts, coarsen(path)])
    events = {h: sorted(v) for h, v in events.items() if len(v) >= min_events}
    OUT.write_text(json.dumps(events))
    n_ev = sum(len(v) for v in events.values())
    vocab = {t for v in events.values() for _, t in v}
    print(f"W1: {len(events)} hosts (>= {min_events} events), {n_ev} events, "
          f"alphabet {len(vocab)} -> {OUT}  (unparsed lines: {bad})")


if __name__ == "__main__":
    main()
