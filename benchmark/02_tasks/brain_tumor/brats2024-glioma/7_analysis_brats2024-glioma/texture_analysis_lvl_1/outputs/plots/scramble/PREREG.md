# Pre-registration: scale-band scrambling, the reliance spectrum (visual proof v14, FINDINGS 41) — written 2026-10-14 before any extraction

Question (Paul): how much does each model RELY on texture (fine-scale information), measured with an intervention that is fair to all arms.
Intervention: decompose the patch image into scale bands with Gaussian low-passes (sigma in voxels): fine = x - G1(x), mid = G1 - G2, coarse = G2 - G4,
base = G4(x). For band b, replace x_b inside the body by a PHASE-RANDOMISED copy (same Fourier amplitude spectrum, uniformly random phases, Hermitian-
symmetric so the result is real; computed on the patch; masked to the body; the band's energy inside the body is rescaled to its original value), keep
the other bands: x^(-b) = x - x_b + scr(x_b). Energy and spectrum are preserved, only the INFORMATION at that scale is destroyed; no mask, no label.
Three random seeds averaged. Also "all-fine" = fine + mid scrambled together.
Arms noise (rung 4, lblvor) / flat (4.5) / real (5), folds 0-2 (fold-mean softmax), v10 patch per case, BraTS first (4 cells, all 70 cases), then
the other 19 cells if the BraTS result is clear.
Outcomes per case, arm, band: target Dice on the patch and J = mean target log-odds over GT target voxels; reliance r_b = Dice(x) - Dice(x^(-b))
(absolute) and r_b / Dice(x) (relative). Gradient link: first-order prediction of the J change, <G, x^(-b) - x> with G = fold-mean gradient of J
w.r.t. the input on the natural image, vs the measured J change (Spearman over cases, per arm and band).

Predictions (Wilcoxon over cases, p < 0.01):
- P1 (reliance ordering): fine-band scrambling costs the real-fill model more Dice than the flat-fill model (r_fine real > flat) in >= 3 of the 4 BraTS
  cells; the coarse and base bands do not show this ordering (real - flat not significant or reversed), i.e. the difference is specific to fine scale.
  Kill: r_fine real <= flat in >= 3 cells.
- P2 (noise arm): the noise-fill model's r_fine is <= the flat model's (it was trained on iid-noise interiors; fine-scale information is not a cue).
- P3 (gradient link): the first-order prediction correlates with the measured J change (Spearman > 0.5) for fine-band scrambling in the real arm;
  if it does not, the gradient-based sensitivity results (sections 33-37) do not predict reliance and this is stated.
- QC: phase scrambling of the fine band must leave the GT-boundary edge energy (|grad G2(x)| on the boundary shell) within [0.8, 1.2] of the original;
  otherwise fine-band scrambling is also a boundary edit and the result is read as "fine-scale appearance (texture + fine boundary detail)".
