# D5 = gigi sessions — next-action prediction results

The PST-family predictors + the Hedge+Fixed-Share **Router** mix from the companion paper,
applied to **gigi's own behavioural data**. Two streams, online/prequential protocol (one model
accumulates across sessions in time order with a depth-bounded context; the Router resets per
session). Stdlib-only, local, no LLM at run time. Data: `results/benchmark_D5-*.json`.

- **D5-event**: per-session CLAIMS event-type sequences (`claim/release/note/progress/...`), 98 sessions.
- **D5-tool**: per-session transcript tool-call sequences (`Bash/Read/Edit/Agent/...`), 3932 sessions / 83 484 positions / 41-tool alphabet.

## Results (acc@1 / acc@5 / nll bits / latency)

| stream | markov0 (floor) | pst | ppm | ctw | lz | **router-soft** | **router-hard** |
|--------|-----------------|-----|-----|-----|----|-----------------|-----------------|
| **D5-event** acc@1 | 0.348 | 0.504 | 0.505 | 0.483 | 0.460 | 0.519 | **0.528** |
| **D5-tool**  acc@1 | 0.460 | 0.550 | 0.552 | 0.552 | 0.521 | **0.564** | 0.563 |
| D5-tool nll  | 2.51 | 2.35 | 2.04 | **1.97** | 2.46 | **1.86** | 1.90 |
| D5-tool lat ms | 0.003 | 0.017 | 0.034 | 0.039 | 0.009 | 0.115 | 0.098 |

## Findings

1. **gigi's next action is predictable from history — and cheaply.** On both streams the
   variable-order Markov predictors beat the most-frequent floor by a wide margin: D5-event
   0.35 → 0.50, D5-tool 0.46 → 0.55 acc@1.
2. **The "mix of predictors" wins.** The Hedge+Fixed-Share **Router** beats the best single predictor
   on both streams (D5-event 0.528 vs 0.505; D5-tool 0.564 vs 0.552), with the best nll
   (D5-tool 1.86 bits). The paper's MoE-style combination is the right read-out for gigi.
3. **Tool-calls are more predictable than session-events** (0.55 vs 0.50 acc@1): the operational
   workflow (Bash→Read→Edit…) is more structured than the coordination verbs.
4. **Latency is microseconds to ~0.1 ms** — three to six orders of magnitude below an LLM
   (~seconds). This is the complement to the membench finding: where the LLM was weak, expensive,
   and model-dependent on next-action, a dedicated PST/Router predictor is accurate, ~free, and
   interpretable — on gigi's own data.

## Protocol note

Online/prequential, depth-bounded (`depth=6`). This differs from the paper's per-session
refit-on-history (`fit(prior)`): there, the model is rebuilt from full prior sequences each
session — O(N²), and it does not scale to D5-tool's 3932 sessions. The online accumulating protocol
is O(total positions), and is also the live-use condition the downstream uses (d/c/b) need. The two
protocols are NOT equivalent (`fit()` builds full-order contexts; online updates a depth-bounded
context) — online gives somewhat higher accuracy here. Reproducible via a stable `(first_ts, key)`
session ordering.

## Sub-line d — anomaly detector (`scripts/run_d5_anomaly.py`)

Reuses the validated Router-mix: at each position the realised action's **nll** (= surprise) is
scored; actions above a bits threshold are flagged, after a `warmup` (cold-start, where everything
is unseen, is not an anomaly). On real D5-tool (warmup 200, threshold 12 bits): **64 anomalies over
3934 sessions**. The most surprising actions are the **first-ever appearances of new capabilities** —
`Skill`, `Workflow`, the chrome MCP tools, `Monitor`, `TaskStop`, `StructuredOutput`, `ExitPlanMode`.
Interpretable novel-action detection: the predictor flags when gigi does something it has never done
in this context. (Refinement for later: separate floored novel-action surprises from
high-but-finite *unusual transitions* between known actions.)

## Roadmap status

- **P1 D5-event** ✓ · **P2 Router-mix** ✓ · **P3 D5-tool** ✓ · **P4 RESULTS** ✓ · **sub-line d anomaly** ✓.
- Next sub-lines (reuse the validated predictor): **c** pre-warming (predict next tool → preload),
  **b** TUI suggester (top-k next actions in-session).
