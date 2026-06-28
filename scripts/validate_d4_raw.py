"""Validate the produced D4 USER*.data files against the same predicate
used for D3 acquisition: ≥5 distinct users with ≥1 user holding ≥50
timestamp-ordered events. D4 is stronger by design (all 12 users have
≥69 events per construction; we keep the predicate for symmetry with
D3 and to fail fast if cohort_selection.txt is hand-edited later).

Exit 0 = PASS. Exit 2 = FAIL.

Usage: python3 scripts/validate_d4_raw.py
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

# Allow importing src/ from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_utils import load_D4  # noqa: E402


def main() -> int:
    try:
        users = load_D4()
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        return 2
    n_users = len(users)
    per_user_events = {
        uid: sum(len(s) for s in sessions) for uid, sessions in users.items()
    }
    max_events = max(per_user_events.values()) if per_user_events else 0
    vocab = Counter()
    for sessions in users.values():
        for sess in sessions:
            vocab.update(sess)
    print(f"D4 users: {n_users}")
    print(f"D4 events per user (sorted desc):")
    for uid, n in sorted(per_user_events.items(), key=lambda kv: -kv[1]):
        print(f"  {uid}: {n}")
    print(f"D4 alphabet |Σ|={len(vocab)}: {dict(vocab.most_common())}")
    ok_users = n_users >= 5
    ok_max = max_events >= 50
    if ok_users and ok_max:
        print(f"\nPASS: {n_users} users, max events/user = {max_events}")
        return 0
    print(f"\nFAIL: users>=5? {ok_users} ; max_events>=50? {ok_max}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
