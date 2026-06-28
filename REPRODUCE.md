# Reproduction guide

Environment: Python 3.9.6, standard library only (no GPU). `pytest` for
tests. All latencies reported on one Apple M-series CPU core,
single-threaded.

## 1. Datasets

### D1 — UCI Unix User Data (real, 9 users)
Source: <https://archive.ics.uci.edu/static/public/141/unix+user+data.zip>
Extract the `USER*` files (gunzip) into `data/D1-shell/USER<k>.data`.
Loader: `src/data_utils.py:load_D1` (`**SOF**`/`**EOF**` session
delimiters; bare command names, args/flags stripped).

### D3 — Eclipse UDC (real IDE, 12-user cohort)
Source: <https://archive.eclipse.org/projects/usagedata/filtered_usagedata_200912.csv.gz>
(filtered public Usage Data Collector, December 2009 export; headerless
positional CSV `userId,what,kind,bundleId,bundleVersion,description,time`).
Gate + build (provenance recorded in `data/D3-eclipse/_raw/SOURCE.md`):
```
python3 scripts/validate_d3_raw.py data/D3-eclipse/_raw/filtered_usagedata_200912.csv.gz
python3 scripts/build_d3_eclipse.py data/D3-eclipse/_raw/filtered_usagedata_200912.csv.gz
```
Cohort = 12 most-active workspaces (>=50 `kind=command` events),
command id normalised to trailing dotted segments, 30-min idle-gap
sessionisation, |Sigma|=105. Deterministic.

## 2. Benchmark runs (regenerates every table)
```
python3 run_benchmark.py --dataset D1 --models pst ppm lz
python3 run_benchmark.py --dataset D1 --models react --max-steps-llm 30
python3 run_benchmark.py --dataset D3 --models pst ppm lz ctw
python3 run_benchmark.py --dataset D3 --models react --max-steps-llm 30 --max-users-llm 5
python3 run_benchmark.py --dataset D1 --compare
python3 run_benchmark.py --dataset D3 --compare
```
LLM model id: `claude-haiku-4-5` via the `claude -p` CLI (OAuth, no API
key). `--max-users-llm` cost-caps the LLM rows; the result JSON doubles
as a resumable checkpoint (per-fold).
Outputs: `results/benchmark_<ds>_<model>.json`,
`results/comparison_<ds>.json`.

## 3. Paper statistics (Table in §5.2)
```
python3 scripts/stats_paired.py
```
Prints bootstrap-95%CI + Holm-corrected Wilcoxon for every PST-vs-X pair.

## 4. Tests
```
python3 -m pytest tests/ -q
```

## Determinism
D3 seed-free (deterministic from the public export + recorded cohort
params). Classical models deterministic. LLM rows depend on the pinned
model id; per-fold acc@1 may vary slightly across LLM runs (reported as
a zero-shot cost reference, not a precise capability number).
