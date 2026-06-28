# ctrouter-probe — GO/NO-GO decision — 2026-05-19

**Probe:** online per-user router (Hedge + Fixed-Share) over a
heterogeneous pool {PST, PPM, CTW, LZ, Markov-1/2/3} vs **static CTW**
(DR-2 bar). D3 only (CTW infeasible on D1 → DR-2 bar undefined there;
gate needs ≥1 dataset; documented deviation). 12-fold LOUO, zero LLM
cost, ~minutes CPU.

## Result (D3, 12 folds, `results/ctrouter_probe.json`)

| reference / candidate | acc@1 | Δ vs static-CTW | folds > CTW |
|---|---|---|---|
| static CTW (bar) | 0.3513 | — | — |
| best fixed expert (PST) | 0.3611 | +0.0098 | — |
| per-user best-expert oracle | 0.3826 | +0.0313 | — |
| any-expert-correct oracle | 0.5223 | +0.1710 | — |
| **router soft** (global, η=0.1, α=0.05) | **0.3876** | **+0.0363** | **11/12** |
| router hard top-1 (ctx, η=1.0, α=0.05) | 0.3610 | +0.0097 | 7/12 |

## **GATE = GO (soft router)**, with an explicit caveat on the sparse variant.

Gate criteria (spec §5):
- router_soft Δ vs static-CTW = **+0.0363 ≥ +0.01** ✓
- fold-consistent: **11/12** folds > CTW (majority) ✓
- any-expert oracle (0.5223) ≫ static-CTW (0.3513) — pool diverse
  enough that routing *can* help ✓
- **Beyond the bar:** the soft router also beats the best fixed expert
  (PST 0.3611) by +0.0265 and slightly exceeds the per-user
  best-single-expert oracle (0.3826) → the gain comes from *adaptively
  combining* experts online per user, not from picking one. This clears
  the honesty bar ("not merely context-mixing re-derived": it beats
  both the static mixture CTW and the best fixed expert, fold-consistent).

**Honest caveat — the MoE-faithful sparse variant is weak.** Hard top-1
(the sparse-activation analogue, the actual long-term MoE goal) is only
Δ +0.0097 vs CTW and 7/12 folds — marginal and not fold-consistent. The
existence signal is in the **soft online per-user mixture**, NOT (yet)
in sparse activation. Any escalation toward a real MoE must specifically
close this soft→sparse gap; it cannot be assumed.

## Phenomenon: exists (soft). Decision: proceed to confirmation, not to real MoE.

This is the first positive probe in the arc (after G1 hybrid NO-GO and
ALT-PROBE MFW NO-GO). Per spec DR-4 + gate discipline, **HALT here**.
Next step is **Approach B** — the confirmatory stress: inject controlled
per-user drift (+1 synthetic stream) to test that the gain is
specifically the *drift-tracking* (Fixed-Share α>0) and not a static
artefact (note: best soft cfg has α=0.05 > 0, consistent with a
drift-tracking effect — suggestive, needs B to confirm vs static CTW
under known drift). Real-MoE escalation only if B confirms AND the
soft→sparse gap is addressed. Do not start B or real-MoE without
explicit user go.
