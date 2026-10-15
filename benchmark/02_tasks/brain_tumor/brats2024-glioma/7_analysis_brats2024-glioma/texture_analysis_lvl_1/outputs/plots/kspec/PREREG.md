# Visual proof v19 (FINDINGS 46): weight-level evidence of fine-scale kernel content (pre-registration)

Written 2026-10-14 ~23:20 MDT (Vulcan clock), BEFORE any kernel spectrum of the 15 settings below or any Part B forward pass was computed.
Script: `benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_kspec.py`;
launchers `run_visual_proof_kspec_vulcan_cpu.sh` (Part A) and `run_visual_proof_kspec_tamia.sh` (Part B), same dir.
Design basis: `LITERATURE_TEXTURE_EVIDENCE_FORMS.md` B2 (permutation-invariant per-layer kernel distributions on the 3D DCT-II basis, random-init
control), B4 (channel-selective weight intervention with a behavioural readout). Prior result this extends: `outputs/plots/weights_analysis/weights_analysis.md`
A1 (BraTS first conv only; post hoc Nyquist band: noise > real > flat ~ baseline, real vs flat separated in both training contrasts).

Wording rule: "high-order kernel energy" / "fine-scale kernel content", never "texture detectors". Instance norm follows every conv, so kernel scale is
meaningless: every kernel is unit-normalised before any spectral quantity.

## Definitions (both parts)

- Kernel basis: orthonormal DCT-II along each axis (`scipy.fft.dctn(k, type=2, norm="ortho")`), so a unit-norm kernel's 27 squared coefficients sum to 1.
  Order of atom (i, j, l) = i + j + l, 0..6 for a 3x3x3 kernel. Anisotropic 1x3x3 kernels (stage 0 of chaos t1in; stages 0-1 of chaos t2spir and ispy2 t2w;
  any other setting whose plans say so) are handled by the same rule (size-1 axis contributes index 0 only): 9 atoms, orders 0..4.
- Per kernel: order profile p_o = energy fraction at order o; **HOF = sum_{o >= 3} p_o** (high-order fraction); **1 - DC** = 1 - p_0.
- iid random kernels have an expected energy of 1/n_atoms per atom, so the random-init order profile must be FLAT across atoms: expected HOF = 17/27 = 0.630
  for 3x3x3 and 3/9 = 0.333 for 1x3x3. This is checked first (sanity gate A0) before any trained-network number is read.
- Near-dead kernels: ||k|| < 1e-6 x the layer's median kernel norm are skipped and counted.
- Layers: every conv weight with a 3x3x3 or 1x3x3 kernel in the encoder (`encoder.stages.*`) and decoder (`decoder.stages.*`). EXCLUDED: the 1x1x1
  segmentation heads, the transposed convs (kernel = stride, 2-tap per axis: the order grouping does not apply) and the duplicate aliases
  (`decoder.encoder.*`, `*.all_modules.*`). Kernels of a layer = all (out, in) pairs.
- Per layer x network: mean HOF over kernels (PRIMARY), median HOF, mean order profile (7 bins), mean 1 - DC.
- Stage groups: **c1** = encoder stage 0 conv 1 (32 kernels, 1 input channel); **c2** = encoder stage 0 conv 2; **E1..E5** = encoder stages 1-5 (mean of
  the two convs' layer values); **D4..D0** = decoder stage at the resolution of encoder stage r (mean of its two convs).

## Part A: kernel spectra of every conv layer, all training settings

Unit of analysis = the TRAINING SETTING (deviation from the task text, which said "23 cells"): the weights do not depend on the test contrast, and the 23
test cells of FINDINGS 37 reuse 15 trained models (counting cells would duplicate identical weights). Settings and the cells they serve:

| setting | arms (noise = lblvor rung 4, flat = flatfill rung 4.5, real = v26_6_2_train050_val000 rung 5) | cells of the 23-cell roster |
|---|---|---|
| glioma_t2w | brats2024-glioma t2w | glioma_t2w_t1n, glioma_t2w_t2f |
| glioma_t1n | brats2024-glioma t1n | glioma_t1n_t2f, glioma_t1n_t2w |
| ms_flair | open-ms flair | ms_flair_t2w, ms_flair_t1w |
| ms_t1w | open-ms t1w | ms_t1w_flair, ms_t1w_t2w |
| brain_t1w | on-harmony T1w | brain_t1w_t2w |
| brain_t2w | on-harmony T2w | brain_t2w_t1w |
| abd_t1in | chaos t1in | abd_t1in_t2spir, abd_t1in_ct |
| abd_t2spir | chaos t2spir | abd_t2spir_t1in, abd_t2spir_t1out, abd_t2spir_ct |
| mand_cbct | toothfairy2 cbct | mand_cbct_hanseg_ct, mand_cbct_pddca_ct |
| breast_t1wce | ispy2 t1wce | breast_t1wce_t2w |
| breast_t2w | ispy2 t2w | breast_t2w_t1wce |
| pelvis_mri | totalseg-pelvic mri | pelvis_mri_ct |
| pelvis_ct | totalseg-pelvic ct | pelvis_ct_mri |
| pancreas_t1wce | pansegdata t1wce | pancreas_t1wce_t2w |
| pancreas_t2w | pansegdata t2w | pancreas_t2w_t1wce |

Run dirs = those of `visual_proof_erf_tasks.CELLS` (flat, real) and `visual_proof_erf_outcome.models` (noise = the single `*lblvor*` sibling);
checkpoint_best.pth, folds 0-2: all 135 present (verified by ls before writing this). Random-init control: 3 networks per SETTING built from that
setting's own plans.json (nnU-Net's He init, torch seeds 0, 1, 2), i.e. 45 random networks.

Statistics.
- Within a setting, fold = replicate (never averaged before a separation test). Raw HOF is compared between arms of the same setting (same architecture).
  **Separated** (stage g, setting s) = all 3 real folds > all 3 flat folds (one-sided exact p = 1/20 = 0.05, the weights_analysis rule). Chance
  expectation 0.05 x 15 = 0.75 separated settings per stage.
- Pooling across settings uses each layer's HOF divided by the same layer's mean HOF over the setting's 3 random inits (**ratio R = HOF / HOF_random**;
  makes 1x3x3 and 3x3x3 layers comparable). Per setting: delta_s(g) = fold-mean R_real - fold-mean R_flat. Across the 15 settings: one-sided Wilcoxon
  signed-rank (alternative real > flat; direction fixed by the BraTS post hoc), sign count, binomial sign test.
- Sensitivity line (not a pass condition): the same test over the 13 non-BraTS settings (the direction was chosen on BraTS).
- 23-cell weighting reported as a count only (for comparability with the grain table), no p-value.

Predictions.
- **PA1** (c1): real > flat by fold mean in >= 10/15 settings AND Wilcoxon over settings p < 0.01.
- **PA2** (c2 AND E1, each by the PA1 rule): fine-scale kernel content carried past the first layer.
- **PA3** (exploratory, no pass/fail): decoder stages D0-D4 show the same ordering.
- **PA4** (reported, not a pass condition): noise arm ordering vs real and flat (noise fill is itself a noise augmentation, expected roughest or smoothest
  for reasons unrelated to lesion texture).
- Kill: PA1 fails -> "no weight-level separation of real vs flat"; reported; Part B runs anyway (independently pre-registered).

## Part B: channel-selective first-stage kernel smoothing with a behavioural readout (BraTS)

Cells glioma_t2w_t1n, glioma_t2w_t2f, glioma_t1n_t2f, glioma_t1n_t2w; arms noise / flat / real; folds 0-2 (fold-mean outputs); ALL 70 held-out cases
per cell (selection is from weights, so no fit set); one patch per case = the v10 edema-centred network patch (`visual_proof_switch.load_case`), natural
images, no input intervention.

Intervention (applied to the weights before the forward pass; restored from a per-fold snapshot after every variant):
- Selection per model (arm x fold) from its OWN weights: conv 1 (encoder stage 0 conv 1, 32 kernels x 1 input): output channel score = HOF of its kernel;
  conv 2 (stage 0 conv 2, 32 x 32): output channel score = mean HOF of its 32 input kernels (near-dead kernels excluded by the Part A rule). Top-k by score.
- **SMOOTH** = project each kernel of a selected output channel onto DCT orders 0-1 (4 atoms), rescale to its original norm. If the projected norm is
  < 1e-6 x the original, the kernel is set to zero (counted).
- Conv sets: c1 (conv 1 only), c2 (conv 2 only), **c12** (both, top-k in each). k in {4, 8, 16}. **Headline: c12, k = 8.**
- Control (i) RANDOM-CHANNEL smoothing: k output channels drawn uniformly without replacement from all 32 (independently in each conv for c12), same
  smoothing; 20 draws per (cell, arm, fold, set, k), seeds hashed on those keys.
- Control (ii) EQUAL-ANGLE random perturbation of the SAME selected kernels: per kernel, theta = angle between k and its smoothed version; k'' =
  ||k|| (cos theta k^ + sin theta v^), v^ uniform random unit vector orthogonal to k^ in the n-atom space. Same norm, same ||k'' - k|| as the smoothing,
  random direction. 20 draws; at k = 8 for c1, c2 and c12 (PB2 is tested at the headline only).
- Outcomes (v17/v18 conventions): J = edema (class 2) log-odds; E = J(natural weights) - J(modified), fold-mean map; scalars = mean over GT edema
  (target), ring 1-10 voxels outside edema (inside body), far background; positive = the modified kernels' removed content supported edema.
  Dice of edema from the fold-mean softmax, before and after (for random draws: mean over draws).
  D_ch = E_sel - mean over draws E_randch; D_pert = E_sel - mean over draws E_pert.
- Plumbing gates: (a) the batched manual stage-0 forward with UNMODIFIED weights reproduces the network's natural logits (max |diff| < 1e-3; fold 0 of
  every arm, every case), and a no-op restore check after the variants of each fold (max |diff| = 0 vs the first natural pass); (b) all-32 smoothing of
  conv 1 changes the target J (reported). Input-blur equivalence check (descriptive, 5 first sorted cases per cell, all arms, fold-mean): correlation over
  body voxels of the edema log-odds map between (1) all 32 conv-1 kernels convolved with a 3x3x3 Gaussian s (sigma 0.75, full 5x5x5 support) vs the
  natural network on the input blurred by s: exact identity up to patch-border padding (plumbing); (2) the same with k*s cropped back to 3x3x3 (Wang
  2020 in-support smoothing); (3) the all-32 DCT order 0-1 projection vs the natural network on the input blurred by a Gaussian with sigma in
  {0.5, 0.75, 1.0, 1.5} (all four reported; none chosen afterwards). (3) is approximate by construction: a DCT projection is not a convolution, plus
  truncation and instance norm.

Predictions (Wilcoxon two-sided over cases, p < 0.01, direction by median sign):
- **PB1**: D_ch (headline) real > flat in >= 3/4 cells (paired real - flat over cases). Real vs noise reported.
- **PB2**: real arm, E_sel > mean E_pert (D_pert > 0) in >= 3/4 cells.
- **PB3**: real arm, Dice drop after selective smoothing (headline) > mean Dice drop after random-channel smoothing in >= 3/4 cells.
- **PB4** (by eye on `kspec_examples_<cell>.png`): the real arm's edema prediction visibly shrinks after selective smoothing and not after random-channel
  smoothing.
- Kill: PB1 fails -> the fine-scale kernel content of the first stage is not used more by the real-fill model; reported and stopped.

Outputs: `outputs/plots/kspec/{PREREG.md, kspec.md, kspec_A_layers.csv, kspec_A_cells.csv, kspec_B_cells.csv, kspec_B_k.csv, kspec_B_controls.csv,
kspec_summary_A.png, kspec_summary_B.png, kspec_firstconv_gallery.png, kspec_examples_<cell>.png (patient slices, never committed), per-case npz/csv}`.
Caveats stated in advance: the noise arm is not neutral; Part A is encode-level (a weight property) unless Part B bites; 3x3x3 kernels have only 7 DCT
orders, so "fine-scale" means 2-3-voxel wavelengths; smoothing changes the network's function beyond texture (the two controls bound this).
