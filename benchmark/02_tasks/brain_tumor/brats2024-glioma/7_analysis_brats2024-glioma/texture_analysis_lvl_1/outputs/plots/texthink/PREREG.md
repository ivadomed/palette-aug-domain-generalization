# Pre-registration: "what the model thinks about texture" — texture-evidence maps by activation patching (visual proof v15, FINDINGS 42), written 2026-10-14 before any extraction

Goal (Paul): open the black box and show that the model THINKS about texture: internal, not Dice. Interventions on the image are allowed; the
readout is the model's own log-odds and its activations.

Counterfactuals (mask-free, label-free, on the v10 patch, fine band = x - G1(x) inside the body):
- x_eq: texture-energy map FLATTENED, every local pattern kept: fine band multiplied by the smooth gain sqrt((m + eps)/(G2(fine^2) + eps)), two passes,
  m = body mean of fine^2 (section 41 follow-up); x_eq = x - fine + fine_eq.
- x_le: fine STRUCTURE scrambled, local energy map kept (phase randomisation + the same local gain matched to the original map; section 41 fine_le).
- QC per case: edge ratio on the GT boundary shell within [0.8, 1.2] for both.

Measurement, per case, arm (noise / flat / real), fold (0-2, fold-mean maps):
1. Texture-evidence map through location L: p_L(v) = sigmoid-free log-odds of the target class at voxel v when the natural image is run with the
   activation at L (skips s0..s5, decoder stages d0..d4) replaced by the counterfactual run's activation (section 39 plumbing). E_L(v) = J_v(x) - p_L(v):
   how much the texture-energy map, as seen at L, pushes the decision at v. Also E_full(v) = J_v(x) - J_v(x_eq) (whole-network counterfactual).
   Same with x_le -> S_L(v), S_full(v) (structure evidence).
2. Scalars: mean of E over GT target voxels and over a peritumoural ring (1-10 vox) and over far background, per L; the fraction of the target voxels
   where E > 1 nat (texture evidence large enough to matter); the ratio E_target / S_target (energy map vs structure).
3. Units: at s2 and s3, per channel c, the relative displacement ||h_c(x_eq) - h_c(x)|| / ||h_c(x)|| over the body; the top-5 channels and the Spearman
   of their natural activation with the input fine-energy map G2(fine^2) (residualised on intensity G2(x) and edge |grad G2(x)|) over body voxels.
   Reports the fraction of channels with displacement > 0.2 per arm.
Visual: 2 cases per cell (median and 75th-percentile edema volume): image | fine-energy map G2(fine^2) | E_full real / flat / noise (same colour
scale) | E_s3 real / flat / noise | top texture unit at s2 (natural activation map) real vs flat. Patient slices: figures not committed.

Predictions (Wilcoxon over cases, p < 0.01):
- P1: E_full over target voxels real > flat in >= 3/4 cells, and real > noise; i.e. the real-fill model's lesion belief depends on the texture-energy map more.
- P2: E_target >> S_target for the real arm (ratio > 2 in >= 3/4 cells): energy map, not pattern.
- P3: the texture evidence enters at s2 / s3 (E_s2 + E_s3 share of E_full > 0.5 for real), consistent with sections 39-40; d3 / d4 are ~1 by construction and not read.
- P4 (units): the fraction of s2/s3 channels with displacement > 0.2 is similar in real and flat (both encode the map; section 31), i.e. the
  difference is in use, not in encoding. If instead real has many more such units, that is reported as the mechanism.
Kill: P1 fails (real <= flat in >= 2 cells) -> the real-fill model's belief is not more texture-map dependent; no further mechanism claim.
