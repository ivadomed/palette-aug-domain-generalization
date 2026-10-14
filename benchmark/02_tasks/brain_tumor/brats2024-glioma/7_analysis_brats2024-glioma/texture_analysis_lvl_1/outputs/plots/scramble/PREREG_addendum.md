# Addendum to PREREG.md (scale-band scrambling, visual proof v14, FINDINGS 41) — written 2026-10-14 BEFORE running the variants below

Status of this addendum: none of the following was in PREREG.md. Both items were set after the first smoke tests, on the coordinating session's instruction:
(1) the local-energy-matched scrambles fine_le / mid_le / allfine_le and the choice of fine_le as the PRIMARY (structure-only) reading of P1-P3 (set after the
2-case smoke test, before the 4-cell run; local gain sigma 2, applied twice); (2) the energy-map flattening variants and predictions P4 / P5 below, set
AFTER the 4-cell result was seen (global fine: real > flat 4/4; fine_le: mixed). They test the explanation suggested by that result, so P4 / P5 are
confirmatory for a post-hoc hypothesis, not part of the original design.

Hypothesis: the real-fill model reacts to the SPATIAL MAP of fine-scale energy (which global scrambling flattens over the body), not to fine-scale structure.
Section 39 showed that scaling the fine band by a constant (relative map kept) moves all arms alike.

Variants (fine and allfine only; same body mask and global body-energy rescale as all other bands):
- fine_eq: the ORIGINAL fine band times the smooth gain g = sqrt((m + eps) / (G_2(b^2) + eps)), re-estimated and applied twice, m = mean of the band^2 inside
  the body, eps = 1e-3 m: local fine energy made spatially uniform over the body, every local pattern (phase) kept. Structure kept, map flattened. Deterministic (1 seed).
- fine_eq_scr: the same flattening applied to the phase-scrambled fine band (same phases as fine / fine_le). Structure destroyed, map flattened. 3 seeds.
- with fine (structure destroyed, map partly flattened by spreading energy evenly) and fine_le (structure destroyed, map kept) the two factors are crossed.
- allfine_eq, allfine_eq_scr: the same for fine + mid.
QC: edge ratio on the GT boundary shell as before; target texture ratio (expected != 1 by design, reported); texture contrast = mean fine energy
(x - G1 x)^2 in the target / mean in the body, before (tc0) and after (tc1): should go to ~1 after flattening.

Predictions (Wilcoxon over cases, p < 0.01, 4 BraTS cells, 70 cases each):
- P4: r_fine_eq real > flat in >= 3/4 cells, and real > noise in those cells. Kill: real - flat on fine_eq not significant in >= 2 cells.
- P5 (the factors add up; flattening explains the global-scrambling effect): fine_eq_scr ~ fine. Operational: in a cell P5 holds if the real - flat
  contrast on fine_eq_scr does not differ from the real - flat contrast on fine (per-case difference of the two contrasts, Wilcoxon p >= 0.01) AND the real -
  flat contrast on fine_eq_scr is itself significantly positive. Supported if it holds in >= 3/4 cells. Reported alongside: per arm, the median of
  r(fine_eq_scr) - r(fine), and the additivity residual r(fine_eq_scr) - [r(fine_le) + r(fine_eq)].

QC note appended after the 2-case smoke (before the 4-cell addendum run; the design above is unchanged): fine_eq moves the target texture contrast toward 1
but not to ~1 (glioma_t1n_t2f, 4 cases: 0.56 -> 0.79, 0.29 -> 0.61, 0.26 -> 0.59, 1.31 -> 1.13), while fine_eq_scr reaches 0.88-1.25. Re-running the
flattening with 4, 8 or 16 iterations instead of 2 changes these by <= 0.03, so the gain has converged; the residual comes from the QC metric
(high-pass x - G1 x of the counterfactual) seeing a different fraction of the fine band in the target than in the body when the band's structure is
kept. The run uses 2 iterations as specified; fine_eq therefore flattens the map only partly as measured by this metric.
