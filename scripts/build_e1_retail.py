"""Build E1 (UCI Online Retail) per-customer timestamped event streams for the
cross-domain timing replication. action token = StockCode; ts = InvoiceDate (s).
Writes data/E1-retail/events.json: {customer_id: [[ts_seconds, stockcode], ...]}."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
XLSX = ROOT / "data" / "E1-retail" / "_raw" / "Online Retail.xlsx"
OUT = ROOT / "data" / "E1-retail" / "events.json"


def main(min_events=20):
    df = pd.read_excel(XLSX)
    df = df.dropna(subset=["CustomerID"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df = df[df["StockCode"].str.len() > 0]
    df["ts"] = (pd.to_datetime(df["InvoiceDate"]).astype("int64") // 10**9).astype(int)
    events = {}
    for cust, g in df.groupby("CustomerID"):
        g = g.sort_values("ts")
        seq = [[int(t), sc] for t, sc in zip(g["ts"], g["StockCode"])]
        if len(seq) >= min_events:
            events[str(int(cust))] = seq
    OUT.write_text(json.dumps(events))
    n_ev = sum(len(v) for v in events.values())
    print(f"E1: {len(events)} customers (>= {min_events} events), {n_ev} events -> {OUT}")
    # quick alphabet stat
    vocab = {sc for v in events.values() for _, sc in v}
    print(f"   distinct StockCodes (alphabet): {len(vocab)}")


if __name__ == "__main__":
    main()
