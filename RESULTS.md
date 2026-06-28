# Results — experiment by experiment

Every number in the paper, the command that produces it, and a one-line reading.
All experiments: **leave-one-user-out (LOUO)**, depth **D = 6**, metrics
**acc@1 / acc@5 / MRR / NLL(bits)**. Paired tests: Wilcoxon signed-rank,
10k-bootstrap 95% CI, Holm across corpora for the confirmatory family, TOST for
equivalence. JSON outputs land in `results/`.

## Corpora

| ID | Source | Domain | \|Σ\| | Users/folds | Events |
|---|---|---|---|---|---|
| D4 | WebLINX (Lù et al. 2024) | web-agent actions | 7 | 12 (leave-one-demonstration-out) | 1,054 |
| D3 | Eclipse UDC | IDE commands | 105 | 12 | ~3.1k |
| D1 | UCI UNIX shell | shell histories | 1,514 | 9 | ~139k |

One action (`click`) is ~60% of D4 — a tiny, mode-dominated alphabet. D1 is the
large, realistic case.

---

## C1 — the router vs the CTW family
`python3 scripts/route_grid_metrics.py` (D3, D4) · `run_d1_fold.py` + `aggregate_d1.py` (D1)

Best-soft-cell acc@1 of the router against each static CTW and the best fixed
model:

| Corpus | router | CTW leaf-KT | CTW faithful-bin | CTW faithful-*M* | best fixed |
|---|---|---|---|---|---|
| D4 | 0.594 | 0.559 | 0.585 | 0.561 | 0.609 (Markov-1) |
| D3 | 0.388 | 0.349 | 0.364 | 0.321 | 0.364 (Markov-1) |
| D1 | 0.242 | 0.211 | 0.193 | 0.179 | 0.224 (PST) |

Paired vs the **faithful-binary** CTW (confirmatory family, Holm across corpora):
D3 Δ=+0.024 (*p*ₕ=0.024, 9/12), D1 Δ=+0.049 (*p*ₕ=0.024, 9/9), D4 Δ=+0.008
(tie, *p*ₕ=0.508). The router also reaches the **lowest NLL on every corpus**
(gain is calibration, not just top-1). The soft mixture drives it: the hard
read-out (single highest-weight expert) is weaker everywhere (0.353 / 0.562 /
0.227).

**Reading:** the router never loses to the CTW family and beats the strongest
static CTW on the medium and large alphabets; the advantage grows with \|Σ\|.

## C2 — which CTW are we beating? read-out vs construction
(same driver; variants in `src/baselines/ctw_faithful.py`)

Holding the construction fixed, the full mixture read-out beats the leaf
read-out on D3 (0.364 vs 0.349) and D4 (0.585 vs 0.559) **but not on D1**
(leaf-KT 0.211 > faithful-bin 0.193). A single *M*-ary tree is much worse
(over-smoothing at large \|Σ\|). So the **strongest static CTW is
corpus-dependent**: faithful-binary on D3/D4, leaf-KT on D1. We use the
faithful-binary as the confirmatory baseline because it is the one CTW variant
**not in the pool** (external comparison), and report that the router beats the
per-corpus strongest CTW anyway (D3 +0.024, D1 +0.031 vs leaf-KT, D4 tie).

## Pool ablation — which experts carry the gain
`python3 scripts/pool_ablation.py` · weights: `scripts/weight_trace.py`

| Corpus | full pool | − CTW | Markov-only |
|---|---|---|---|
| D3 | 0.387 | 0.374 (−0.013) | 0.366 (−0.021) |
| D4 | 0.592 | 0.601 (+0.010) | 0.602 (+0.010) |

**Reading:** on the larger alphabet the variable-order experts (CTW above all)
carry the gain; on the tiny D4 a first-order Markov chain already suffices.
Mean weights confirm the split of labour (CTW backbone, PPM/Markov-1 local
corrections on D3).

## Robustness — the 18-cell grid
`route_grid_metrics.py` reports all 18 (η, α, scope) cells. Every cell beats
leaf-KT and faithful-*M* on D3 and D4 (18/18); against faithful-binary the
count is 14/18 (D3), 9/18 (D4), 18/18 (D1). Even the **worst** cell clears
leaf-KT (D3 0.354 vs 0.349, D4 0.571 vs 0.559) — the headline does not rest on
a tuned cell. A single fixed config (η=1, α=0.05, global) clears faithful-binary
on D3 (+0.014) and D1 (+0.046), and by a hair on D4 (+0.007).

## vs the deployer's pragmatic fallback (Markov-1) — TOST
At n=12 the router-vs-Markov-1 comparison is **inconclusive** on D3 (Δ=+0.024,
*p*=0.155) and D4 (Δ=−0.015, *p*=0.131): neither significant nor TOST-equivalent
(90% TOST CI [+0.0035,+0.0469] on D3, [−0.0286,−0.0020] on D4 — both outside
±0.02). On the large D1 the pool beats Markov-1 (+0.038, *p*=0.021, 8/9) and the
best fixed expert PST (+0.018, 9/9). Markov-1 was picked *post hoc* — these are
winner's-curse contrasts, reported as descriptive.

## C3 — three negative mechanism probes (what the gain is *not*)
Each fixes a kill criterion in advance and returns NO-GO.

| Probe | Script | Result | Verdict |
|---|---|---|---|
| **P1** not drift-tracking | `drift_stream.py`, `ctrouter_probe_B.py` | Fixed-Share contribution Δα=+0.003 vs +0.02 gate | NO-GO |
| **P2** not antidictionary gating | `mfw_probe.py`, `personal_antidict_probe.py` | strictly worse than plain PST on 12/12 folds (population *and* personal-online MFW; deterioration grows cold→late) | NO-GO |
| **P3** not a zero-shot LLM hybrid | `hybrid_probe.py`, `hybrid_bounds.py`, `llm_react.py` | on D3 high-entropy steps a zero-shot `qwen2.5:14b` reaches acc@1 0.27 vs 0.48 for Markov-1 | NO-GO |

**Reading:** the gain is generic adaptive mixing over a heterogeneous pool, not
a nameable mechanism — which keeps the deployment-rule claim honest.

## §neural — a learned baseline, and the pool that absorbs it
`scripts/neural_baseline.py` (GRU out of pool, role A) ·
`scripts/neural_pool.py` (GRU in pool → router-8, role B) ·
model: `src/baselines/neural_gru.py` (GRU4Rec-style, one fixed config across
corpora, trained once per fold and frozen, same depth-6 context).

| Corpus | GRU | router-7 (classical) | router-8 (pool+GRU) | router-8 − router-7 |
|---|---|---|---|---|
| D4 | 0.628 | 0.593 | **0.637** | +0.044 (*p*=0.008, 9/12) |
| D3 | 0.269 | 0.387 | **0.387** | −0.000 (*p*=0.86) |
| D1 | 0.256 | 0.242 | **0.274** | +0.032 (*p*=0.008, 9/9) |

GRU out of pool (role A): wins D4 (0.628 vs 0.593, *p*=0.041), loses D3 (0.269
vs 0.388, *p*=0.010 — data-starved at 105 symbols over ~3k events, the regime
where simple baselines keep pace with deep models), numerically ahead but not
significant on D1 (0.256 vs 0.242, *p*=0.17).

**Reading:** no single model wins across the alphabet range. Add the GRU to the
pool (router-8) and the router uses it where it helps (D4, D1) and ignores it
where it hurts (D3) — best or tied everywhere, beating even the GRU alone. The
deployment rule absorbs a learned expert untuned: *only the pool grows*.

---

### Determinism
`random.seed(0)` / `torch.manual_seed(0)` at module entry; per-fold
determinism is documented in each `results/*DECISION.md`. The classical pool is
pure-Python (stdlib); the GRU baseline needs `torch` (CPU is enough).
