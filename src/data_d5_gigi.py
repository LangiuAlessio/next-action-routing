"""D5 = gigi sessions. D5-event: per-session CLAIMS event-type sequences, for temporal
next-action prediction. Stdlib-only; gigi data stays local."""
import json
import os

_PRUNE = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}


def load_D5_event(root, min_seq_len=3):
    by_session = {}  # session -> list[(ts, event)]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE]
        if "CLAIMS.jsonl" not in filenames:
            continue
        with open(os.path.join(dirpath, "CLAIMS.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sess, ev, ts = r.get("session"), r.get("event"), r.get("ts")
                if not (sess and ev and ts):
                    continue
                by_session.setdefault(sess, []).append((ts, ev))
    out = {}
    for sess, evs in by_session.items():
        evs.sort(key=lambda x: x[0])
        seq = [e for _, e in evs]
        if len(seq) >= min_seq_len:
            out[sess] = (evs[0][0], seq)
    return out


def load_D5_tool(root, min_seq_len=3):
    """D5-tool: per session-transcript the ordered tool-call names (from assistant tool_use
    blocks). root = a dir of `*.jsonl` Claude Code transcripts (recursive, skipping `memory`).
    One file = one session; key = filename stem, first_ts = first record timestamp."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (_PRUNE | {"memory"})]
        for fn in filenames:
            if not fn.endswith(".jsonl"):
                continue
            tools, first_ts = [], None
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if first_ts is None and r.get("timestamp"):
                        first_ts = r["timestamp"]
                    if r.get("type") != "assistant":
                        continue
                    content = (r.get("message") or {}).get("content")
                    if isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name"):
                                tools.append(b["name"])
            if len(tools) >= min_seq_len and first_ts:
                out[fn[:-6]] = (first_ts, tools)  # strip ".jsonl"
    return out


def temporal_split(dataset):
    items = sorted(dataset.items(), key=lambda kv: (kv[1][0], kv[0]))  # by first_ts
    splits = []
    for i, (key, (_ts, seq)) in enumerate(items):
        train = [v[1] for _, v in items[:i]]  # prior sessions' sequences
        splits.append((key, train, [seq]))
    return splits
