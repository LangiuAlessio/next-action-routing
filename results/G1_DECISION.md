# G1 decision — hybrid feasibility probe — 2026-05-19

**Gate rule (plan):** OPEN iff, for ≥1 dataset, oracle gain ≥ +0.03
acc@1 at call-rate ≤ 0.20 **AND** realistic gain ≥ 0.00 (hybrid ≥ PST).

## Result

| | PST base | oracle @≤0.20 | realistic @≤0.20 |
|---|---|---|---|
| D1 | 0.2260 | 0.3400 (**+0.114**) | 0.2034 (**−0.023**) |
| D3 | 0.3755 | 0.4776 (**+0.102**) | 0.3397 (**−0.036**) |

- Oracle condition: **PASS** (huge headroom — the entropy gate *does*
  isolate hard steps; a perfect LLM on the top-~15% entropy steps would
  lift D1 0.226→0.34, D3 0.376→0.48).
- Realistic condition: **FAIL on both, at every call-rate** (the
  measured zero-shot ReAct ~9% is so weak that routing high-entropy
  steps to it *lowers* accuracy below pure PST).

## **G1 = CLOSED.**

The entropy-gated hybrid **with the zero-shot LLM is dead** (it makes
accuracy worse, not better). The cheap probe (zero LLM cost, ~7 min
compute) has saved the 6–9-week Phase 2–6 effort.

## Actionable break-even (does a *fair* LLM (M5) revive it?)

Hybrid beats pure PST iff the gated-step LLM accuracy exceeds **PST's
own accuracy on the high-entropy gated steps**:

| call-rate | D1 break-even | D3 break-even |
|---|---|---|
| ~5% | 20.2% | 36.1% |
| ~10% | 23.5% | 34.7% |
| ~15% | 24.0% | 32.4% |
| ~20% | 23.2% | 30.8% |

A fair (few-shot/long-context) LLM would need to score **~24% on D1 /
~33% on D3 on exactly the steps where PST is most uncertain** (novel /
post-behaviour-change contexts — the intrinsically hardest). Published
next-action LLM accuracy (Fu et al. 2025) is ~11–12% *overall*; clearing
~24–33% on the hardest subset is implausible. **Conditional revival is
unlikely; treat the hybrid (M1) as not viable.**

## Consequence

The hybrid (M1) is not pursued further.
