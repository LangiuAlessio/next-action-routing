"""
Dataset loading and preprocessing for D1 (Unix shell) and D2 (mobile app launches).

D1 — UCI Unix User Data:
    88 users, ~shell command sequences from .bash_history.
    Download: https://archive.ics.uci.edu/dataset/95/unix+user+data
    Format: one command per line, blank line = session boundary.

D2 — Mobile app launch surrogate:
    If Falaki 2010 is not directly downloadable, we generate a synthetic
    surrogate with realistic statistical properties (Zipf distribution
    over app names, geometric session length, per-user Markov chains).
    Replace load_D2() with real data when available.
"""

from __future__ import annotations
import os
import re
import random
import math
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
D1_DIR = DATA_DIR / "D1-shell"
D2_DIR = DATA_DIR / "D2-mobile"


# ---------------------------------------------------------------------------
# D1 — Unix shell commands
# ---------------------------------------------------------------------------

def load_D1(
    raw_dir: Path = D1_DIR,
    min_seq_len: int = 3,
) -> dict[str, list[list[str]]]:
    """
    Load UCI Unix User Data (9 users, Purdue, up to 2 years of tcsh history).
    Format: **SOF**/**EOF** session delimiters; <N> = sanitised arguments (skipped);
    flags (-x), metacharacters (| & > <) kept as separate tokens but we keep
    only bare command names for the action alphabet.

    Returns dict {user_id: [session_sequence, ...]} where each sequence is a
    list of command-name strings (no arguments, no flags).
    """
    user_files = sorted(raw_dir.glob("*.data"))
    if not user_files:
        raise FileNotFoundError(
            f"D1 data not found in {raw_dir}.\n"
            "Download: https://archive.ics.uci.edu/static/public/141/unix+user+data.zip\n"
            "Extract USER*.gz files (gunzip) into data/D1-shell/ as USER*.data"
        )

    users: dict[str, list[list[str]]] = {}
    for fpath in user_files:
        uid = fpath.stem          # USER0, USER1, …
        sessions: list[list[str]] = []
        current: list[str] = []
        in_session = False

        for raw_line in fpath.read_text(errors="replace").splitlines():
            token = raw_line.strip()
            if token == "**SOF**":
                in_session = True
                current = []
            elif token == "**EOF**":
                if in_session and len(current) >= min_seq_len:
                    sessions.append(current)
                in_session = False
                current = []
            elif not in_session:
                continue
            else:
                cmd = _normalise_cmd_d1(token)
                if cmd:
                    current.append(cmd)

        if sessions:
            users[uid] = sessions
    return users


# Skip sanitised arguments (<N>), flags (-x), metacharacters, empty tokens.
_SKIP_D1 = re.compile(r"^(<\d+>|-.*|[|&<>{}()\[\]!@#$%^*]+)$")


def _normalise_cmd_d1(token: str) -> Optional[str]:
    """Return bare command name or None to skip this token."""
    if not token:
        return None
    if _SKIP_D1.match(token):
        return None
    # take only the base name in case a path slipped through
    cmd = os.path.basename(token)
    # must start with a letter or digit
    if not cmd or not (cmd[0].isalpha() or cmd[0].isdigit()):
        return None
    return cmd.lower()


D3_DIR = DATA_DIR / "D3-eclipse"
D4_DIR = DATA_DIR / "D4-weblinx"


def _normalise_cmd_d3(token: str) -> Optional[str]:
    """D3 tokens are already normalised command-ids written by
    build_d3_eclipse.normalise_cmd_id; keep as-is, drop empties."""
    t = token.strip()
    return t.lower() if t and t[0].isalnum() else None


def load_D3(
    raw_dir: Path = D3_DIR,
    min_seq_len: int = 3,
) -> dict[str, list[list[str]]]:
    """Eclipse UDC IDE-command workflow (D3). Same on-disk format
    (**SOF**/**EOF**) and return contract as load_D1:
    dict {user_id: [session_seq(list[str]), ...]}."""
    user_files = sorted(raw_dir.glob("*.data"))
    if not user_files:
        raise FileNotFoundError(
            f"D3 data not found in {raw_dir}.\n"
            "Run: python3 scripts/build_d3_eclipse.py "
            "data/D3-eclipse/_raw/<file>"
        )
    users: dict[str, list[list[str]]] = {}
    for fpath in user_files:
        uid = fpath.stem
        sessions: list[list[str]] = []
        current: list[str] = []
        in_session = False
        for raw_line in fpath.read_text(errors="replace").splitlines():
            tok = raw_line.strip()
            if tok == "**SOF**":
                in_session = True
                current = []
            elif tok == "**EOF**":
                if in_session and len(current) >= min_seq_len:
                    sessions.append(current)
                in_session = False
                current = []
            elif not in_session:
                continue
            else:
                c = _normalise_cmd_d3(tok)
                if c:
                    current.append(c)
        if sessions:
            users[uid] = sessions
    return users


def load_D4(
    raw_dir: Path = D4_DIR,
    min_seq_len: int = 3,
) -> dict[str, list[list[str]]]:
    """WebLINX agent-action sequences (D4). Same on-disk format
    (**SOF**/**EOF**) and return contract as load_D1/load_D3:
    dict {user_id: [session_seq(list[str]), ...]}.

    Per the framing-shift declared in the paper §1+§4 and in
    data/D4-weblinx/_raw/SOURCE.md, the "user" unit here is one
    stable WebLINX demonstration (no per-annotator id exists).
    Action tokens are BrowserGym function names: click, say,
    text_input, scroll, load, submit, change (|Σ|=7). Each USER<k>
    file holds exactly one demonstration as a single session.
    Cohort: top-12 demos by event count (see cohort_selection.txt).
    """
    user_files = sorted(raw_dir.glob("*.data"))
    if not user_files:
        raise FileNotFoundError(
            f"D4 data not found in {raw_dir}.\n"
            "Run: python3 scripts/build_d4_weblinx.py "
            "(after scripts/explore_d4_weblinx.py)"
        )
    users: dict[str, list[list[str]]] = {}
    for fpath in user_files:
        uid = fpath.stem
        sessions: list[list[str]] = []
        current: list[str] = []
        in_session = False
        for raw_line in fpath.read_text(errors="replace").splitlines():
            tok = raw_line.strip()
            if tok == "**SOF**":
                in_session = True
                current = []
            elif tok == "**EOF**":
                if in_session and len(current) >= min_seq_len:
                    sessions.append(current)
                in_session = False
                current = []
            elif not in_session:
                continue
            else:
                c = _normalise_cmd_d3(tok)  # alphanumeric tokens, same rule
                if c:
                    current.append(c)
        if sessions:
            users[uid] = sessions
    return users


def _normalise_cmd(raw: str) -> Optional[str]:
    """Legacy helper kept for compatibility."""
    raw = raw.strip().lstrip("$ ")
    if not raw or raw.startswith("#"):
        return None
    token = re.split(r"[\s;|&<>]", raw)[0]
    return os.path.basename(token) if token else None


# ---------------------------------------------------------------------------
# D2 — Mobile app launches (synthetic surrogate)
# ---------------------------------------------------------------------------

_DEFAULT_APPS = [
    "whatsapp", "instagram", "chrome", "maps", "gmail", "spotify",
    "youtube", "camera", "photos", "settings", "calendar", "clock",
    "telegram", "twitter", "netflix", "uber", "amazon", "paypal",
    "slack", "zoom", "files", "contacts", "phone", "messages",
    "news", "weather", "fitness", "banking", "notes", "calculator",
]

_INTENT_GROUPS = {
    "communication": ["whatsapp", "telegram", "twitter", "slack", "zoom",
                      "gmail", "phone", "messages"],
    "entertainment": ["instagram", "youtube", "spotify", "netflix", "news"],
    "productivity":  ["chrome", "maps", "calendar", "notes", "files",
                      "calculator", "contacts"],
    "services":      ["uber", "amazon", "paypal", "banking", "weather",
                      "fitness", "camera", "photos", "settings", "clock"],
}

ACTION_TO_INTENT: dict[str, str] = {
    app: intent
    for intent, apps in _INTENT_GROUPS.items()
    for app in apps
}


def generate_D2_synthetic(
    n_users: int = 50,
    sessions_per_user: int = 20,
    session_len_mean: int = 8,
    zipf_alpha: float = 1.2,
    intent_switch_prob: float = 0.15,
    seed: int = 42,
) -> dict[str, list[list[str]]]:
    """
    Synthetic mobile app-launch dataset with per-user preferences and
    realistic intent-switching behaviour.

    Each user has a Zipf-distributed preference over apps (personalized
    ranking), with intent clusters shaping the distribution.
    """
    rng = random.Random(seed)
    intents = list(_INTENT_GROUPS.keys())

    def zipf_weights(apps: list[str], alpha: float) -> list[float]:
        n = len(apps)
        ws = [1.0 / (i + 1) ** alpha for i in range(n)]
        total = sum(ws)
        return [w / total for w in ws]

    users: dict[str, list[list[str]]] = {}
    for u in range(n_users):
        uid = f"user_{u:03d}"
        # per-user shuffle of apps within each intent group
        user_order = {
            intent: rng.sample(apps, len(apps))
            for intent, apps in _INTENT_GROUPS.items()
        }
        sessions: list[list[str]] = []
        for _ in range(sessions_per_user):
            seq: list[str] = []
            intent = rng.choice(intents)
            for _ in range(max(3, int(rng.gauss(session_len_mean, 2)))):
                if rng.random() < intent_switch_prob:
                    intent = rng.choice(intents)
                apps_ordered = user_order[intent]
                weights = zipf_weights(apps_ordered, zipf_alpha)
                app = rng.choices(apps_ordered, weights=weights, k=1)[0]
                seq.append(app)
            sessions.append(seq)
        users[uid] = sessions
    return users


def intent_labels_D2(sequences: list[list[str]]) -> list[list[int]]:
    """Convert app-launch sequences to intent label sequences (int index)."""
    intent_map = {k: i for i, k in enumerate(_INTENT_GROUPS.keys())}
    return [
        [intent_map.get(ACTION_TO_INTENT.get(a, ""), -1) for a in seq]
        for seq in sequences
    ]


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def leave_one_user_out(
    users: dict[str, list[list[str]]],
) -> list[tuple[str, list[list[str]], list[list[str]]]]:
    """
    Yield (test_user, train_sequences, test_sequences).
    Train = all sequences from all other users (flattened).
    Test = sequences of test_user.
    """
    all_users = list(users.keys())
    result = []
    for test_uid in all_users:
        train = [seq for uid, seqs in users.items() if uid != test_uid for seq in seqs]
        test = users[test_uid]
        result.append((test_uid, train, test))
    return result
