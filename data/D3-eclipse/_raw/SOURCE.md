# D3 raw data provenance — Eclipse UDC (filtered usage data)

- **Dataset:** Eclipse Usage Data Collector (UDC), filtered usage data
  (only `org.eclipse.*` bundles).
- **Source URL:** https://archive.eclipse.org/projects/usagedata/filtered_usagedata_200912.csv.gz
- **Access date:** 2026-05-19
- **File chosen:** December 2009 (`200912`) — the smallest monthly dump
  (227 MB gzip) of the 2009-01..2010-08 archive.
- **Acquired:** **prefix of 111,427,269 / 237,821,745 bytes** of the gzip
  stream. The Eclipse archive mirror served at ~76 KB/s; the full file
  did not complete in a 15-min window. The partial gzip is used as a
  **prefix-of-December-2009** sample. This is sound for a benchmark
  cohort (we sample a fixed cohort of active users, not a
  month-complete population). `validate_d3_raw.parse()` tolerates the
  truncated tail (keeps all records read before truncation).
- **Schema (verified against the file, headerless positional CSV):**
  `userId, what, kind, bundleId, bundleVersion, description, time(ms epoch)`.
  Real user actions are rows with `kind == "command"`; `description`
  holds the Eclipse command id; `userId` is a per-workspace UUID.
- **Gate result (2026-05-19):** `{"n_users": 60000 (cap), "users_ge_50ev": 4672}`
  → PASS (threshold: ≥5 users, ≥1 user with ≥50 command events).
- **Citation (for references.bib, Task C3):** Eclipse Foundation, Usage
  Data Collector, filtered usage data archive
  (https://archive.eclipse.org/projects/usagedata/). A peer-reviewed
  companion citation (e.g., Murphy-Hill, Parnin & Black, "How We
  Refactor, and How We Know It", ICSE 2009 / TSE 2012, which analysed
  UDC data) to be confirmed at Task C3.
- **Committed here:** `validate_d3_raw.py`, this `SOURCE.md`,
  `sample.csv` (first 200 raw rows). The full `*.csv.gz` is gitignored
  (size); rebuild by re-downloading the URL above.
