# D4 source — WebLINX-BrowserGym (Route b empirical extension)

## Acquisition decision (Phase 2 gate PASS with caveats)
**Date:** 2026-05-20.
**Chosen candidate:** Priority 3 of the spec shortlist (LLM-agent
trajectory logs) — **WebLINX-BrowserGym v1.1**, after Priority 1
(modern shell history) was found to have no clean public per-user
behavioural dataset and Priority 2 (IDE telemetry) was skipped on prior
(mostly proprietary).

**Dataset URL:** https://huggingface.co/datasets/McGill-NLP/weblinx-browsergym
**GitHub:** https://github.com/McGill-NLP/WebLINX
**Paper:** https://arxiv.org/abs/2402.05930
**License:** **CC BY-NC-SA 4.0** (NonCommercial + ShareAlike) — Hugging
Face dataset card. Repository code itself is Apache-2.0; the *data* is
CC BY-NC-SA 4.0 (this is the binding license for our use).

## License fence (important for the Zenodo artefact at Phase 6)
- Use in academic paper + cite: permitted.
- Derived per-user `USER<k>.data` files extracted from WebLINX: must be
  released **under CC BY-NC-SA 4.0** (ShareAlike), not the bench's
  default MIT. The Phase-6 Zenodo bundle must split:
  - bench code, scripts, D1, D3, paper artefacts → MIT (unchanged).
  - D4 derived per-user files → CC BY-NC-SA 4.0 in a subfolder with
    its own LICENSE-derived file + citation to McGill-NLP/WebLINX.
- Document this fence in the bench `LICENSE`/`README` at Phase 6.

## Framing shift (acknowledged in the paper §1 and §4)
WebLINX is **human web-navigation demonstrations** (not bash/IDE
commands). The Route b paper's "digital workflows" framing expands to
include "agent action sequences" in §1 and §4 with one sentence
acknowledging this is a different action vocabulary (web actions:
click/type/scroll/tab-action over DOM elements) but the same
next-action-prediction task (LOUO over demonstrators, online update,
acc@1). The C1 claim becomes "router beats the depth-D leaf-KT VOMM
family on three real workflow corpora spanning shell, IDE, and web
agent demonstrations".

## Acquisition strategy (Task 2.2 — pending)
WebLINX is 140 GB total. We will NOT download the full dataset.
Plan:
1. Use the HuggingFace `datasets` library in **streaming mode** to
   inspect schema (10 records) — confirms `Demonstration`/`Turn`
   structure (per the GitHub README; full schema TBC by inspection).
2. Stream + filter to a manageable cohort: select the top-N
   demonstrators by number-of-demonstrations (where "demonstrator" is
   the stable id field — TBC by inspection; if absent, treat
   "demonstration_id" as the unit and do leave-one-demonstration-out
   per the spec § honest-framing-shift).
3. Materialise to `data/D4-weblinx/USER<k>.data` in the D1
   on-disk format (`**SOF**` ... action tokens one per line ...
   `**EOF**`) using a Task-2.2 builder `scripts/build_d4_weblinx.py`.
4. Action token = the discrete BrowserGym action type (click, type,
   scroll, tab-action, etc.) — normalised to a stable string, alphabet
   ~10-20 tokens (expected, TBC by inspection).
5. Standard D3-pattern validator `scripts/validate_d4_raw.py` predicate:
   ≥5 distinct users/demonstrators with ≥50 events each.

## Schema items still to confirm in Task 2.2 (hands-on inspection)
- Exact field name for stable user/demonstrator identifier (vs
  demonstration/session id alone). If only `demonstration_id` is
  stable per record, LOUO becomes leave-one-demonstration-out
  (documented framing shift).
- Action type vocabulary size (rough).
- How turns map to per-step (one turn = one action, or one turn =
  multiple actions).

## Honest risks
- The 140-GB full corpus is overkill; streaming subset risks
  partial-coverage bias. Mitigation: document the cohort selection
  rule in the paper (same as D3 cohort=top-12-by-event-count).
- License complications around the Zenodo bundle. Mitigation:
  split the bundle with explicit per-folder licensing (see fence
  above).
- Action vocabulary smaller than D1/D3 (likely 10-20 vs 100+).
  Implication: CTW is feasible (no infeasibility issue as on D1);
  PPM/PST should also be cheap. Smaller alphabet may compress
  the relative differences between predictors.

## Gate status
**D4-ACQ = PASS** (with the documented framing shift + license fence).
Phase 2 conceptually closed; **Task 2.2 (builder) + 2.3
(loader+wiring) deferred to next session** — they are non-trivial work
(HF streaming, schema inspection, builder) that benefits from a
dedicated chunk of focused execution, not a tail-end of a long session.
