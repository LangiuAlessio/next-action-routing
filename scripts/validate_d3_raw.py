"""Validate a raw Eclipse-UDC filtered dump can yield per-user temporal
command sequences, and parse it.

Usage: python3 scripts/validate_d3_raw.py <raw_path(.csv|.csv.gz)>
Exit 0 = PASS (proceed Task A1). Exit 2 = FAIL (unusable / aggregate-only).

Contract: parsed records expose (user_id, command_id, timestamp);
>= 5 distinct users; >= 1 user with >= 50 timestamp-ordered command events.

REAL SCHEMA (verified against archive.eclipse.org/projects/usagedata/,
filtered_usagedata_YYYYMM.csv.gz, 2026-05-19): the filtered CSV has
**no header**; columns are positional:

    0 userId   1 what   2 kind   3 bundleId   4 bundleVersion
    5 description   6 time(ms since epoch)

Only rows with kind == "command" are real user actions (the rest is
bundle/perspective startup noise); description holds the command id.
This module's parse() is the SINGLE point of raw-format coupling
(build_d3_eclipse imports it). Streaming + gzip-aware + memory-bounded
so multi-GB monthly dumps do not OOM.
"""
import sys
import csv
import gzip
import json
import io
from collections import defaultdict
from pathlib import Path

# Memory/runtime bounds (a monthly UDC dump is millions of rows).
_MAX_USERS = 60_000
_MAX_EV_PER_USER = 6_000
_MAX_CMD_ROWS = 6_000_000


def _open(raw_path: Path):
    if str(raw_path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(raw_path, "rb"), errors="replace")
    return open(raw_path, "r", errors="replace")


def parse(raw_path: Path):
    """Return {user_id: [(ts_seconds:int, command_id:str), ...]} for
    kind=="command" rows only. ms epoch -> integer seconds (so the
    downstream idle-gap sessioniser works in seconds as planned)."""
    users = defaultdict(list)
    seen_cmd = 0
    # Tolerate a TRUNCATED gzip (interrupted download): keep every record
    # read before the truncation; the trailing gzip/CRC/EOF error is
    # expected and non-fatal for a prefix-of-month dataset.
    try:
        with _open(raw_path) as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 7:
                    continue
                if row[2].strip() != "command":
                    continue
                seen_cmd += 1
                if seen_cmd > _MAX_CMD_ROWS:
                    break
                uid = row[0].strip()
                cmd = row[5].strip()
                ts_raw = row[6].strip()
                if not uid or not cmd or not ts_raw:
                    continue
                try:
                    ts_s = int(int(ts_raw) // 1000)  # ms -> s
                except ValueError:
                    continue
                if uid not in users and len(users) >= _MAX_USERS:
                    continue
                ev = users[uid]
                if len(ev) < _MAX_EV_PER_USER:
                    ev.append((ts_s, cmd))
    except (EOFError, OSError, gzip.BadGzipFile, csv.Error):
        pass  # truncated stream -> use the prefix read so far
    return users


def main():
    if len(sys.argv) != 2:
        print("usage: validate_d3_raw.py <raw_path>", file=sys.stderr)
        sys.exit(2)
    raw = Path(sys.argv[1])
    if not raw.exists():
        print(json.dumps({"error": f"not found: {raw}"}))
        sys.exit(2)
    users = parse(raw)
    n_users = len(users)
    rich = sum(1 for evs in users.values() if len(evs) >= 50)
    print(json.dumps({"n_users": n_users, "users_ge_50ev": rich}))
    sys.exit(0 if (n_users >= 5 and rich >= 1) else 2)


if __name__ == "__main__":
    main()
