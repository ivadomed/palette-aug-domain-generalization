# Pre-registration: the texture switch (visual proof v12, FINDINGS 39) — written 2026-10-14 before any extraction

Scope: BraTS only (glioma_t2w_t1n, glioma_t2w_t2f, glioma_t1n_t2f, glioma_t1n_t2w), arms noise (rung 4, lblvor) / flat (4.5) / real (5), folds 0-2,
all 70 (69) test cases; plan patch around the target (edema, label 2) as in v10. Script `scripts/visual_proof_switch.py`, run on TamIA.
Counterfactual: x_a = x + (a - 1) highpass_1(x) inside the body (mask-free, label-free, not a pure scaling). Erasure directions are computed on the
first 20 cases (sorted) per cell from h(x_2) - h(x_0.5) and evaluated on the other 50 (49).

Predictions (decision rule: Wilcoxon over cases, p < 0.01, same sign of the median in the three folds where fold-level numbers exist):
- P1 (dose-response). Real-fill Dice falls on BOTH sides of a = 1 (a = 0.5 and a = 2) and falls more than flat-fill on both sides; flat-fill changes
  little (|median change| < 0.03 at a = 2); noise-fill falls at a = 0.5 and not at a = 2. Kill: real does not fall at a = 2 (then "real fill dislikes
  smooth input" is not excluded), or flat falls as much as real.
- P2 (mediation). In the real-fill model the counterfactual change of J is carried mainly by the high-resolution skips s0 / s1 and by the last decoder
  stages d3 / d4 (sum of their single-location shares > 0.5 of dJ_full); in the flat-fill model dJ_full is small (|dJ_full| < half of real's).
  Kill: shares spread evenly or concentrated in the deep skips s4 / s5.
- P3 (the switch). Projecting the rank-4 texture subspace out of the real-fill model at d4 (pre-registered location; d3 and s0 as secondary) on
  untouched images lowers its edema Dice toward the flat-fill model's natural Dice (median drop > 0.05 AND drop larger than with a random rank-4
  subspace, p < 0.01), while the same erasure in the flat-fill model moves its Dice by < 0.03 and random subspaces move real by < 0.03.
  If rank 4 is not enough but rank 16 is, report the rank (secondary outcome, weaker claim). Kill: real-fill Dice does not drop beyond the random
  control at any location / rank, or the flat-fill model drops as much.
- QC gate: at a = 2 the GT-boundary edge energy ratio must stay within [0.8, 1.6] and the within-target texture energy ratio within [1.5, 2.5]
  (the edit is a texture edit, not a boundary edit); otherwise the dose-response is re-run with a smaller a range before reading P1.
