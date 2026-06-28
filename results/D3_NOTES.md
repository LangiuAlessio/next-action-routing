# D3 (Eclipse UDC) run notes — 2026-05-19

- Source: prefix of filtered_usagedata_200912.csv.gz (see data/D3-eclipse/_raw/SOURCE.md).
- Cohort: 12 users, min 50 command events, 30-min idle-gap sessionisation.
- Full-corpus |Sigma| = 105 distinct normalised command-ids.
- CTW: **FEASIBLE on D3**, wall ~285 s for full LOUO (well under the
  25-min/fold infeasibility cutoff used on D1). This DEVIATES from the
  spec's predicted "CTW infeasible at large Sigma": at |Sigma|=105 CTW
  completes. Scalability boundary observed: CTW feasible at |Sigma|=30
  (old D2) and |Sigma|=105 (D3), infeasible at |Sigma|~400 (D1).
  => CTW finding is NOT one-sided; reflect this honestly in the paper
     (supersedes spec section 5's "one-sided" assumption — data-driven).
- Models ran: PST, PPM, CTW, LZ (classical, complete). ReAct N1: Phase B.
