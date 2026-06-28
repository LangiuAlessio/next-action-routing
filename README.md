# Reproducibility bundle — *Online Routing for Next-Action Prediction*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20995956.svg)](https://doi.org/10.5281/zenodo.20995956)


Companion artefact (C4) to the paper **Online Routing for Next-Action
Prediction** (A. Langiu, G. Pilato; under submission to DMKD). Source: the
`cdt-prediction-bench` repository. This bundle is the deposit-ready snapshot;
mint the Zenodo DOI from `make_bundle.sh`'s output once the paper is sign-off
ready, and insert the DOI into the paper (§ contributions, C4) and the cover.

## What the paper claims, and where it lives here

| Paper element | In the bundle |
|---|---|
| Pool: PST, PPM, CTW leaf-KT, Lempel-Ziv, Markov-1/2/3 | `src/baselines/{pst,ppm,ctw,lz_pred}.py`, `scripts/markov_k.py` |
| Three static CTW variants (leaf-KT, faithful binary, faithful M-ary) — C2 | `src/baselines/{ctw,ctw_faithful}.py` (faithful binary is the confirmatory baseline) |
| Hedge + Fixed-Share router (soft/hard) — C1 | `scripts/ctrouter.py` |
| Full 18-cell grid + paired metrics (acc@1/5, MRR, NLL) | `scripts/route_grid_metrics.py`; D1 per-fold `scripts/run_d1_fold.py` + `aggregate_d1.py` |
| Pool ablation + weight traces | `scripts/pool_ablation.py`, `scripts/weight_trace.py` |
| Three negative mechanism probes (P1 drift, P2 antidictionary, P3 LLM) — C3 | `scripts/{drift_stream,ctrouter_probe_B,mfw_probe,personal_antidict_probe,hybrid_probe,hybrid_bounds,llm_react}.py`; decisions in `results/*DECISION.md` |
| Learned GRU baseline + router-8 (pool + GRU) — §neural | `src/baselines/neural_gru.py`, `scripts/neural_baseline.py`, `scripts/neural_pool.py`; `results/neural_baseline.json`, `results/neural_pool.json` |
| Paired statistics (bootstrap CI, Wilcoxon, Holm, TOST) | `scripts/stats_paired.py`, helpers in `route_grid_metrics.py` |
| Corpus builders (D1 shell, D3 Eclipse UDC, D4 WebLINX) | `src/data_utils.py`, `scripts/build_d3_eclipse.py`, `scripts/build_d4_weblinx.py` (+ validators) |
| Seeds / determinism | `random.seed(0)` / `torch.manual_seed(0)` at module entry; per-fold determinism documented in each DECISION file |

## Reproducing the headline (acc@1, leave-one-user-out, depth 6)

```
# classical pool + router grid + static CTW baselines (D3, D4)
python3 scripts/route_grid_metrics.py
# D1 (large alphabet): per-fold then aggregate
for k in $(seq 0 8); do python3 scripts/run_d1_fold.py --fold $k; done
python3 scripts/aggregate_d1.py
# learned baseline and router-8 (needs torch; CPU is enough)
python3 scripts/neural_baseline.py --corpora D4 D3 D1   # GRU out of pool (role A)
python3 scripts/neural_pool.py --corpus D4              # GRU in pool, router-8 (role B)
python3 scripts/neural_pool.py --corpus D3
for k in $(seq 0 8); do python3 scripts/neural_pool.py --corpus D1 --fold $k; done
python3 scripts/neural_pool.py --corpus D1 --aggregate
```

Environment: Python 3.12 (3.9+ for the classical pool, stdlib only); the GRU
baseline needs `torch` (CPU). See `REPRODUCE.md` for the dataset rebuild from
public sources.

## Licensing (fence)

- Code, scripts, results, decisions: **MIT** (`LICENSE`).
- Derived **D4 (WebLINX)** corpus files: **CC BY-NC-SA 4.0**, inherited from
  `McGill-NLP/WebLINX` (Lù et al. 2024) by ShareAlike (`data/D4-weblinx/LICENSE`).
- **Raw** source datasets (UCI Unix .gz, Eclipse UDC .gz, WebLINX parquet) are
  **not** redistributed; rebuild deterministically from the public sources per
  the `SOURCE.md` files. D1/D3 derived files are public-research-use.

## Building the deposit archive

`bash make_bundle.sh` assembles a clean, license-fenced archive
(`compact-trie-bench-bundle/` + `.zip`) from the `cdt-prediction-bench`
checkout, excluding raw data and build artefacts. Upload the `.zip` to Zenodo
with `.zenodo.json` as metadata, then record the DOI.
