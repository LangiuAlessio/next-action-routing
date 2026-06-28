import json
from data_d5_gigi import load_D5_event, load_D5_tool, temporal_split


def test_load_D5_tool_extracts_toolcalls(tmp_path):
    rec = [
        {"type": "system", "timestamp": "2026-06-01T10:00:00.000Z"},
        {"type": "assistant", "timestamp": "2026-06-01T10:00:01.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "ok"}, {"type": "tool_use", "name": "Bash"}]}},
        {"type": "user", "timestamp": "2026-06-01T10:00:02.000Z"},
        {"type": "assistant", "timestamp": "2026-06-01T10:00:03.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "name": "Read"}, {"type": "tool_use", "name": "Edit"}]}},
    ]
    (tmp_path / "sess1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rec))
    (tmp_path / "short.jsonl").write_text(
        json.dumps({"type": "assistant", "timestamp": "2026-06-02T10:00:00Z",
                    "message": {"content": [{"type": "tool_use", "name": "Bash"}]}}) + "\n")
    d = load_D5_tool(str(tmp_path))
    assert set(d) == {"sess1"}                              # short.jsonl dropped (1 tool < 3)
    first_ts, seq = d["sess1"]
    assert seq == ["Bash", "Read", "Edit"]                  # in record order
    assert first_ts == "2026-06-01T10:00:00.000Z"



def _claims(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_load_D5_event_groups_and_orders(tmp_path):
    _claims(tmp_path / "leafA" / "CLAIMS.jsonl", [
        {"ts": "2026-06-02T10:00:00+00:00", "session": "s1", "event": "release", "target": "A"},
        {"ts": "2026-06-01T10:00:00+00:00", "session": "s1", "event": "claim", "target": "A"},
        {"ts": "2026-06-01T11:00:00+00:00", "session": "s1", "event": "note", "target": "A"},
    ])
    _claims(tmp_path / "leafB" / "CLAIMS.jsonl", [
        {"ts": "2026-06-05T10:00:00+00:00", "session": "s2", "event": "claim", "target": "B"},
    ])  # s2 has only 1 event -> dropped by min_seq_len=3
    d = load_D5_event(str(tmp_path))
    assert set(d) == {"s1"}                                  # s2 dropped (too short)
    first_ts, seq = d["s1"]
    assert seq == ["claim", "note", "release"]               # ordered by ts
    assert first_ts == "2026-06-01T10:00:00+00:00"


def test_temporal_split_orders_and_accumulates():
    ds = {"s2": ("2026-06-05", ["claim", "release"]),
          "s1": ("2026-06-01", ["claim", "note", "release"]),
          "s3": ("2026-06-09", ["claim", "progress"])}
    splits = temporal_split(ds)
    assert [k for k, _, _ in splits] == ["s1", "s2", "s3"]    # by first_ts
    assert splits[1] == ("s2", [["claim", "note", "release"]], [["claim", "release"]])
    assert splits[0][1] == []                                # first is cold-start (no history)
