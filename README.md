# Online Routing for Next-Action Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20995955.svg)](https://doi.org/10.5281/zenodo.20995955)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility bundle for the paper **"Online Routing for Next-Action
Prediction"** (A. Langiu, G. Pilato). It is the full benchmark behind every
number in the paper: corpus builders, predictors, the router, the negative
probes, the learned baseline, and the paired-statistics helper.

---

## TL;DR

A deployer who must predict a user's next action (in an IDE, a shell, a
web-agent) has to **pick one predictor before seeing the stream** — and the
best predictor depends on the corpus, which you only learn afterwards. That
choice is a gamble.

We show you can **avoid the choice**: run a small, untuned pool of classical
predictors (PST, PPM, CTW, Lempel–Ziv, Markov-*k*) and let a **regret-bounded
Hedge + Fixed-Share router** re-weight them online after every symbol — no
per-user tuning, no training, a few dozen bytes of state. The contribution is
not a new predictor; it is the **deployment rule**.

Two further results sharpen it:
1. We separate two CTW design choices benchmarks conflate (read-out vs
   construction) and hold the router to the **strongest static CTW**, not a
   weakened default.
2. The pool is **extensible**: drop in a learned GRU expert and the router
   *absorbs* it untuned — helping where it is good, harmless where it is bad —
   so the combined pool is best on every corpus.

## Headline results — acc@1, leave-one-user-out, depth 6

The untuned router vs the strongest static CTW per corpus, and vs the best
single fixed model (the pragmatic "just pick one" choice):

| Corpus | \|Σ\| | router (pool) | strongest static CTW | best fixed model | router − CTW |
|---|---|---|---|---|---|
| D4 (WebLINX web-agent) | 7 | 0.594 | 0.585 (faithful-bin) | 0.609 (Markov-1) | +0.008 (tie, *p*=0.51) |
| D3 (Eclipse IDE) | 105 | **0.388** | 0.364 (faithful-bin) | 0.364 (Markov-1) | **+0.024** (*p*ₕ=0.024) |
| D1 (UNIX shell) | 1514 | **0.242** | 0.211 (leaf-KT) | 0.224 (PST) | **+0.031** vs leaf-KT, *9/9* |

The router **never loses** to the CTW family and **beats the strongest static
CTW on the medium and large alphabets** (Holm-corrected). The advantage grows
with the alphabet. The confirmatory baseline is the faithful-binary CTW — the
one CTW variant the router does *not* contain, so the comparison is external.

## The extensible pool — a learned expert, absorbed (acc@1)

No single model wins everywhere; the router routes to whoever is best.

| Corpus | GRU (out of pool) | router-7 (classical) | **router-8 (pool + GRU)** |
|---|---|---|---|
| D4 | 0.628 (wins alone) | 0.593 | **0.637** |
| D3 | 0.269 (data-starved) | 0.387 | **0.387** |
| D1 | 0.256 | 0.242 | **0.274** |

Adding the GRU helps on D4 (+0.044, *p*=0.008) and D1 (+0.032, *p*=0.008, 9/9)
and is harmless on D3 (the router down-weights it). **router-8 matches or beats
the GRU alone on every corpus** — including D4, where the GRU beats the
classical router. *Only the pool grows; the router stays the same.*

See **[RESULTS.md](RESULTS.md)** for every experiment, command and number.

## What's inside

| Paper element | File(s) |
|---|---|
| Pool: PST, PPM, CTW leaf-KT, LZ, Markov-1/2/3 | `src/baselines/{pst,ppm,ctw,lz_pred}.py`, `scripts/markov_k.py` |
| Static CTW variants (leaf-KT, faithful binary, faithful *M*-ary) | `src/baselines/{ctw,ctw_faithful}.py` |
| Hedge + Fixed-Share router (soft/hard) | `scripts/ctrouter.py` |
| Headline grid + paired metrics | `scripts/route_grid_metrics.py`; D1: `scripts/run_d1_fold.py`, `aggregate_d1.py` |
| Pool ablation / weight traces | `scripts/pool_ablation.py`, `scripts/weight_trace.py` |
| Negative probes P1/P2/P3 | `scripts/{drift_stream,ctrouter_probe_B,mfw_probe,personal_antidict_probe,hybrid_probe,hybrid_bounds,llm_react}.py` |
| Learned GRU baseline + router-8 | `src/baselines/neural_gru.py`, `scripts/neural_baseline.py`, `scripts/neural_pool.py` |
| Paired statistics (bootstrap, Wilcoxon, Holm, TOST) | `scripts/stats_paired.py`, helpers in `route_grid_metrics.py` |
| Corpus builders (D1/D3/D4) | `src/data_utils.py`, `scripts/build_d3_eclipse.py`, `scripts/build_d4_weblinx.py` |

## Reproduce

```bash
# classical pool + router grid + static CTW (D3, D4)
python3 scripts/route_grid_metrics.py
# D1 (large alphabet): per fold, then aggregate
for k in $(seq 0 8); do python3 scripts/run_d1_fold.py --fold $k; done
python3 scripts/aggregate_d1.py
# learned baseline (needs torch, CPU is enough) and router-8
python3 scripts/neural_baseline.py --corpora D4 D3 D1     # GRU out of pool
python3 scripts/neural_pool.py --corpus D4                # GRU in pool (router-8)
python3 scripts/neural_pool.py --corpus D3
for k in $(seq 0 8); do python3 scripts/neural_pool.py --corpus D1 --fold $k; done
python3 scripts/neural_pool.py --corpus D1 --aggregate
```

Python 3.9+ (stdlib only for the classical pool); the GRU baseline needs
`torch` (CPU). Datasets rebuild deterministically from public sources — see
[REPRODUCE.md](REPRODUCE.md) and the `SOURCE.md` files.

## Cite

```bibtex
@misc{langiu_pilato_nextaction_bench,
  author = {Langiu, Alessio and Pilato, Giovanni},
  title  = {Online Routing for Next-Action Prediction --- reproducibility bundle},
  year   = {2026},
  doi    = {10.5281/zenodo.20995955},
  url    = {https://doi.org/10.5281/zenodo.20995955}
}
```

## License

MIT (code, scripts, results) — see [LICENSE](LICENSE). The derived **D4
(WebLINX)** corpus files inherit **CC BY-NC-SA 4.0** from
[McGill-NLP/WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX)
(`data/D4-weblinx/LICENSE`). Raw datasets are not redistributed.
