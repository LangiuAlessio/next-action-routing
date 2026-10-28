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

## Standing of the route

Two new-method levers have now been falsified on this data: the entropy-gated PST⊕LLM hybrid (G1)
and MFW-antidictionary gating (this). There is no in-reach novel predictor that beats PST here.
