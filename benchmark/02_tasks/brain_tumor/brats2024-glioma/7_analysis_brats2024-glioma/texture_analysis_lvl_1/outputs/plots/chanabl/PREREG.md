# Pre-registration: texture-energy-selective CHANNEL ABLATION inside the network (visual proof v18, FINDINGS 45), written 2026-10-14 before any extraction

Goal: dissection-then-lesion (Bau et al. 2017 / 2020 dissection; Morcos et al. 2018 mean-ablation). Find the channels of each stage whose activation
tracks the local texture-energy map, ablate them, and ask whether the real-fill model's edema decision depends on them more than the flat-fill model's,
beyond variance-matched random channels. Follow-up to v17 (FINDINGS 44, LEACE erasure of the 7-dim linear texture-energy subspace).
Wording throughout: "texture-energy-selective channels", never "texture detectors" or "texture pattern detected".

## Prior from v17 (stated honestly, same evidence family)
v17 removed the linearly decodable texture-energy subspace (rank 7) at single stages. At d2: real > flat in 3/4 cells (p <= 5e-8 in those three,
t2w -> t2f +0.09 p 0.02), real > noise in only 2/4 (both T2w-trained cells noise >= real). The texture-specific effect (texture minus equal-rank random)
peaked at d3 in 4/4 cells. Equal-rank random erasure did nothing (|E| <= 0.013 nat). Intensity-subspace erasure collapsed every arm on the FLAIR targets
(5.9-8.4 nat; Dice 0.76 -> 0.29, 0.62 -> 0.08) but barely moved the t1n -> t2w cell (0.4 nat, Dice unchanged). Channel ablation is a COARSER removal of
the same concept (whole channels selected by correlation, not the optimal linear subspace), so the expectation is confirmation where v17 separated real
from flat (t2w -> t1n, t1n -> t2f, t1n -> t2w), not a rescue where it did not (t2w -> t2f; real vs noise in the T2w-trained cells). A positive v18 result
does not overturn v17's kill; a negative one does not overturn v17's real > flat cells.

## Cells, arms, cases (identical to v17)
- Cells: glioma_t2w_t1n, glioma_t2w_t2f, glioma_t1n_t2f, glioma_t1n_t2w; 70 test cases per cell, sorted by name; FIT = first 20, EVALUATION = other 50.
  All effect numbers and tests are on the 50 evaluation cases.
- Arms: noise (rung 4, `*lblvor*`), flat (rung 4.5, `*flatfill*`), real (rung 5, `*v26_6_2_train050_val000*`), folds 0-2; random-init floor "init" = the
  real network re-initialised by `reset_parameters` (torch seed 1000 + k for "fold" k), with its OWN selection on the fit cases.
- Input: the v10 plan patch per case (`visual_proof_switch.load_case`), natural test images only; fp32, no autocast. Patch 128 x 160 x 112.

## Stages and channel counts
Skips s2 (C = 128, 1/4 res), s3 (256, 1/8), s4 (320, 1/16); decoder d1 (256, 1/8), d2 (128, 1/4), d3 (64, 1/2). HEADLINE STAGE d3 (a priori: v17's
texture-specific effect peaked at d3 in 4/4 cells); all stages reported. Semantics identical to v17: skip ablation = SKIP-PATH ablation (only the tensor
concatenated into the decoder; the encoder's own downstream path is untouched); decoder ablation = the stage output (post-nonlinearity, LeakyReLU; the
tensor that feeds the next transposed convolution). Decoder re-entry from the ablated point exactly as v17's `decode_from`.

## Descriptors (image-only, mask-free, label-free; FINDINGS 31, identical to v17)
At full patch resolution: I = G2(x); E = |grad G2(x)|; Tf = sqrt(G2((x - G1 x)^2)); texture concept r = log(Tf + 1e-6) residualised per case on
[1, I, log(E + 1e-6)] over body voxels (OLS). Downsampled to each stage's grid as the body-masked area average; stage body = area(body) >= 0.5.
Spearman is rank-based, so E vs log E is immaterial for the edge selection.

## Channel selection (fit cases only; per cell x arm x fold x stage)
- Per fit case and channel c: Spearman rho_c(case) between the channel's activation and r over the case's stage-body voxels (ranks computed per case;
  ties broken by order, LeakyReLU makes exact ties rare). Selection score = |mean over the 20 fit cases of rho_c(case)| (sign-agnostic, but a channel must
  be sign-consistent across cases to score). Texture-selective top-k = the k channels with the highest score. Intensity-selective (POSITIVE CONTROL) and
  edge-selective (secondary control) = the same with I and E.
- k in {4, 8, 16, 32}, capped at C/2 (never binding here: min C = 64). HEADLINE k = 16.
- Texture units per layer (descriptive by-product, not a test): number of channels whose score exceeds the 99th percentile of a null made by spatially
  SHUFFLING r within each case (20 shuffles; null score per channel and shuffle = |mean over cases of rho with the shuffled map|; p99 over channels x
  shuffles). Known limitation, stated now: the shuffle null ignores spatial autocorrelation, so its p99 is tiny (~1/sqrt(n voxels)) and most channels
  are expected to exceed it; the count is therefore close to C and weakly informative. Added secondary null (my addition, before extraction): CASE-SWAP,
  the channel activation of fit case i correlated with r of fit case i - L (L = 1, 2, 3) over the intersection of the two stage-body masks (same patch
  grid), mean over pairs per lag, p99 over channels x lags; this keeps spatial autocorrelation and anatomy-level structure. Both counts reported per arm and
  stage (mean over folds), plus the Jaccard overlap at k = 16 of the texture-selected set with the intensity- and edge-selected sets (if high, "texture-
  energy-selective" carries less weight than it sounds).
- Per-channel fit-set statistics: mean mu_c and variance over stage-body voxels of the 20 fit cases (pooled).

## Ablation (evaluation cases)
MEAN-ablation (Morcos et al. 2018 convention): the selected channels' activations are replaced by their fit-set mean mu_c at EVERY voxel of the patch, at
inference, per fold with that fold's own selection; the decoder is re-entered from that point. Caveat stated now: every ablated tensor feeds a conv
block whose instance norm removes per-channel constants, so mean- and zero-ablation are near-equivalent here (up to padding / transposed-conv boundary
effects): what is removed is the channel's SPATIAL VARIATION. The Morcos convention is kept for comparability.

## Nulls
- PRIMARY: VARIANCE-MATCHED random channels. Channels of the stage are binned into deciles of fit-set activation variance (by rank). For each selected
  channel, one random channel is drawn, without replacement within the draw, from the NON-selected channels of the same decile; if that decile is
  exhausted, from the nearest decile with a free channel (adjacent first, lower before higher on ties). 20 draws, seeded by md5(cell, arm, fold, stage, k,
  draw). The fallback rate is logged per stage and k (expected non-zero at d3 k = 32, where half of 64 channels are selected).
- Secondary: plain random k non-selected channels (20 draws).
- Reported per case: E_tex - mean over draws of E_matched ("texture minus matched random", D), and E_tex - 95th percentile over the 20 matched draws (with 20
  draws about the 19th-20th order statistic).

## Outcomes (per evaluation case; fold-mean over folds 0-2)
J(v) = edema (class 2) log-odds, fold-mean. PRIMARY: signed target-mean effect E_k(v) = J(v) - J_ablated(v) averaged over GT edema voxels (signed because
v16 showed flat / noise use the same cue with the opposite sign; positive = the ablated channels pushed the edema belief UP). Also: peritumoural ring
(dilation 10 minus target, in body) and far background (body minus dilation 10). SECONDARY: target Dice on the patch (argmax of the fold-mean softmax ==
class 2 vs GT == 2) before and after ablation (texture, intensity, edge, matched draw 0, plain draw 0). Every (stage, k, variant) computed; per-draw
target means stored for the bands.

## Plumbing gate (read before any effect; independent of P3)
(a) Decoder re-entry with ZERO channels ablated reproduces the natural logits (max |diff| ~ 0; smoke and the real arm fold 0 of every run case logged).
(b) Ablating ALL channels of a stage changes the target log-odds by > 1 nat in median and lowers median Dice by > 0.2 in every trained arm at d3 (recorded
per case as variant "all", one per stage). (c) In the smoke cell (glioma_t1n_t2f, a FLAIR target where v17 found intensity dominant) the k = 16 intensity
ablation at d3 must move the target log-odds more than the matched-random p95. If (a) or (b) fails, the plumbing is fixed before reading results.
Non-collapse of the intensity control in the T1-target / T2-target cells would be consistent with v17 (intensity mattered on FLAIR targets only) and is
NOT read as a plumbing failure.

## Predictions (Wilcoxon signed-rank, two-sided, over the 50 evaluation cases per cell, p < 0.01, direction = sign of the median paired difference)
- P1 (headline; d3, k = 16): D = E_tex - mean E_matched, real > flat in >= 3/4 cells. Real vs noise reported, NOT a pass condition (the noise arm is
  itself a noise augmentation and v17 showed it is not a neutral control).
- P2 (real arm, d3, k = 16): E_tex - E_matched_p95 > 0 in >= 3/4 cells.
- P3 (positive control, d3, k = 16): intensity-selected ablation drops median Dice by > 0.2 in EVERY trained arm in 4/4 cells. Given v17, expected to
  hold on the two FLAIR-target cells and possibly to fail on the T1n / T2w targets; a failure there is a result about the edema decision, not a
  plumbing failure (see the plumbing gate).
- P4 (readability, by eye): in the example figure the real arm's edema prediction shrinks visibly after texture-channel ablation (d3, k = 16) and not after
  matched-random ablation (draw 0).
- KILL: P1 fails -> channel ablation does not separate real from flat; report and stop (no new design from these data).
Also reported (not tested as predictions): every stage x k table; D for noise / flat / real; raw E_tex; intensity and edge D; plain-random null;
real - init and the init floor at d3 k = 16; ring and background effects.

## Visuals (patient slices: Vulcan only, NOT committed)
Per cell 2 evaluation cases: closest to the cell median and to the 75th percentile of per-case real - flat D (d3, k = 16); slice = the axial slice with
the most GT edema. Rows real / flat / noise; columns image | prediction (edema, arm colour; GT edema contour lime) | after texture-channel ablation
(d3, k = 16) | after matched-random ablation (draw 0) | effect map E painted on the image (orange = the ablated texture channels supported edema, cyan =
opposed; shared symmetric scale = 99th percentile of |E| over the three arms). Side panels: E vs k at d3 per arm with the matched-random band (2.5-97.5
percentile over draws, median over cases); texture-units-per-layer bars per arm. Summary figure without patient data `chanabl_summary.png`: per cell,
bars of D at d3 k = 16 for the three arms with the intensity control (E_int - mean E_matched) and the init floor; E-vs-k curves; unit counts.

## Outputs
`outputs/plots/chanabl/{PREREG.md, chanabl.md, chanabl_cells.csv, chanabl_depth_k.csv, chanabl_units.csv, chanabl_controls.csv, chanabl_summary.png,
chanabl_examples_<cell>.png (patient data), <cell>/fit/*.npz, <cell>/<case>_chanabl.csv, <case>_chanabl_maps.npz (patient data)}`.

## Smoke (before the run; not read for results)
glioma_t1n_t2f, 3 fit + 2 evaluation cases, fold 0, 2 draws, into `/scratch/p/paulh/toDelete/chanabl_smoke`: plumbing gate (a)-(c), output completeness,
timing per case (to size the run).
