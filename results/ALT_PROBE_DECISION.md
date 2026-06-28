# ALT-PROBE decision — MFW/antidictionary-gated PST — 2026-05-19

**Idea:** alternative novel predictor to replace the dead PST⊕LLM hybrid
(G1 closed): PST positive memory **gated by the minimal-forbidden-word
antidictionary** (forbid actions closing a forbidden factor; force when
one survives — DCA principle). Mignosi-domain → co-author + IEEE-TIT/TCS
lineage. Probe reused `forbidden-factors-exp/src/mff_extract.py` on the
existing D1/D3 LOUO, zero LLM cost.

## Result (`results/mfw_probe.json`)

| dataset | PST | MFW-gated | Δ | gated>PST | forced steps | forced acc |
|---|---|---|---|---|---|---|
| D1 (shell) | 0.2222 | 0.1777 | **−0.0445** | 0/9 | 135 | 0.116 |
| D3 (IDE)   | 0.3590 | 0.2922 | **−0.0668** | 0/12 (1 tie) | 44 | 0.085 |

## **ALT-PROBE = CLOSED.** MFW-gated PST is strictly worse, every fold.

**Why (analytic, not a bug):** under leave-one-user-out the held-out
user emits action n-grams absent from the *other* users' training; those
become "minimal forbidden words" and the gate forbids exactly the
actions the test user legitimately takes. The antidictionary/DCA
prediction principle is provably strong on **structured / low-entropy**
sources (Sturmian, DNA — the in-house dca-sturmian / antidictionary
theory) but cross-user behavioural streams are high-entropy with
per-user drift: the super-critical regime where the
`antidictionary-2026` phase-transition theorem itself predicts no
compression/prediction gain. Hard forcing is especially bad here
(forced-step accuracy ≈ 0.09–0.12).

## Standing of the "novel predictor → Q1" route

Two natural new-method levers have now been cheaply falsified on this
data: the entropy-gated PST⊕LLM hybrid (G1) and MFW-antidictionary
gating (this). The incremental-test discipline did its job (~minutes
each, no weeks lost). The honest position: there is **no in-reach novel
predictor that beats PST here**; the "new method to unlock >65% Q1" plan
is not supported by the data.

### Untested softer variants (NOT pursued without explicit go — avoid spiral)
- MFW as a *soft* re-rank prior on PST top-k (not hard zeroing).
- Antidictionary built **online from the test user's own prefix** (not
  other users' training) — removes the LOUO cross-user artefact; closest
  to a fair "personal antidictionary" and the only variant with a real
  chance, but it changes the method and is a fresh probe.
- High-min-len MFW only (≥4) to reduce false forbids.

### Salvage value
The negative result is itself a clean, honest finding linking the DCA
phase-transition theory to behavioural prediction ("antidictionary
forcing fails on high-entropy personalised streams"). It is a *paragraph
/ related discussion*, not a venue-lifting contribution.

## Recommendation (unchanged from post-G1)
1. **Short paper now** (IUI/UMAP) on the honest v0.10 (Phase-0 polish done).
2. **Benchmark-resource rescope** (no novel predictor needed): M2 DOI +
   M3 modern corpora + M5 fair LLM reference + M6 stats → TKDD/D&B
   (~40–55% / ~30–40%).
Decision for Alessio/Giovanni. Optionally: one more cheap probe of the
*personal-antidictionary* variant only (it is the single variant that
neutralises the LOUO artefact) before fully abandoning the MFW route.
