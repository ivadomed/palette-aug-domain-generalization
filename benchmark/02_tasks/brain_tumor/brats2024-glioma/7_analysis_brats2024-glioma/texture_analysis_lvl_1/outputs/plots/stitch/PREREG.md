# Pre-registration: model stitching, "where does real fill make the difference" (visual proof v13, FINDINGS 40) — written 2026-10-14 before any extraction

Premise (sections 31 / 37-addendum / 39): flat-fill (rung 4.5) and real-fill (rung 5) networks have near-identical representations (linear CKA diverges
in 0/44 stage x contrast cells) and respond alike to fine-detail interventions, yet differ in OOD Dice. Hypothesis: the advantage is carried by a
late re-wiring (decoder tail / readout) over a shared feature set, not by different features.

Design (BraTS cells glioma_t2w_t1n, glioma_t2w_t2f, glioma_t1n_t2f, glioma_t1n_t2w; folds 0-2 paired by fold index; v10 patch per case; natural
test images only, no image intervention):
- Stitch map at a location L: ridge regression W_L, b_L from the SOURCE network's activation at L to the TARGET network's activation at L
  (same architecture, same channel count), voxel-wise, fitted on the body voxels of the first 20 cases (sorted) of the cell, label-free,
  lambda = 1e-3 * trace(X'X)/C. Evaluated on the other 50 (49) cases.
- Hybrids (direction F->R = flat source, real target): E = all six encoder skips mapped, real decoder; D_s (s = 0..4) = flat run up to decoder
  stage s, mapped, real network continues from stage s+1 using REAL's own skips; D_4 = flat d4 mapped -> real seg layer (readout swap).
  Reverse direction R->F likewise. Control: cross-fold self-stitch F_k -> F_k' and R_k -> R_k' (k' = k+1 mod 3) at every location = the
  stitching penalty between networks that differ only by fold.
- Outcome per case: target Dice on the patch (fold-mean softmax over the 3 fold pairs). Fraction of the gap recovered at L (F->R):
  g_L = (Dice_hybrid - Dice_flat) / (Dice_real - Dice_flat), reported where the real - flat gap is > 0.05 Dice (cell medians), else the raw Dice.

Predictions (Wilcoxon over cases, p < 0.01):
- P1 (stitching works): cross-fold self-stitch costs < 0.03 Dice at every location (otherwise the stitch is lossy and results are read with the
  penalty subtracted).
- P2 (late re-wiring): in F->R, the fraction recovered is >= 0.7 at D_4 (readout swap) and at D_3, and the F->R hybrid at D_4 is within 0.03 Dice
  of the real-fill parent. Kill: D_4 recovers < 0.3 of the gap.
- P3 (symmetry): R->F at D_4 loses >= 0.7 of the gap (real features + flat readout ~ flat Dice).
- Secondary (exploratory): if P2 holds, the depth profile g_L (E, D_0 .. D_4) shows where the gap first appears; if E already recovers >= 0.7, the
  advantage is in the encoder and the hypothesis is revised, not rescued.
