# Pre-registration: "what did the model use to predict this pixel" — Integrated Gradients from the texture-flattened twin (visual proof v16, FINDINGS 43), written 2026-10-14 before any extraction

Goal (Paul): a gradient / activation map that shows what the model looked at to call a pixel, with texture made visible; not an output metric.
Baseline: x_eq = the image with its local fine-energy map flattened to the body mean, every local pattern kept (v15 construction, two outer passes;
mask-free, label-free). Path: x(t) = x_eq + t (x - x_eq), t in [0, 1], M = 16 steps (midpoint rule), per fold, fold-mean maps.
Decision: for seed voxel v (section-36 seeds, <= 12 per case, each arm's OWN predicted class as in v10), f_v = log-odds of that class at v.
Attribution map: A_v(u) = (x(u) - x_eq(u)) * mean_t dJ_v/dx(u) at x(t). Completeness check: sum_u A_v(u) = f_v(x) - f_v(x_eq) (logged, relative error).
A_v is non-zero only where the texture-energy map differs from the body mean, so it shows WHICH TEXTURED LOCATIONS the model drew on for v.
Feature-level version at skip s3 (and d2): C_v(u) = sum_c (h_c(x) - h_c(x_eq))(u) * mean_t df_v/dh_c(u) (Grad-CAM-style on the counterfactual path),
upsampled to the patch.
Numbers per seed: mass inside the GT target region, in the peritumoural ring (1-10 vox), far background: m_in = sum A_v over region / sum |A_v| over the
body; signed total f_v(x) - f_v(x_eq); the same for C_v. Per case median over seeds; Wilcoxon over cases per cell; arms noise / flat / real.
Visual: per cell 2 cases (median and 75th percentile target volume), one target seed each (pattern CCC if available, else FFC): image | fine-energy map |
A_v real / flat / noise (shared symmetric scale by the 99th percentile of |A| over the three arms) | C_v at s3 real / flat / noise | seed marked.
Patient slices: not committed.

Predictions (p < 0.01, Wilcoxon over cases):
- P1: for target seeds the signed total (texture-map contribution to the pixel's own-class log-odds) is larger for real than flat in >= 3/4 cells and
  larger than noise in 4/4 (the per-pixel version of section 42).
- P2: the attribution mass inside the target, m_in, is larger for real than for flat in >= 3/4 cells (the real model's texture evidence for a lesion
  pixel comes from the lesion's own texture, not from elsewhere); for the noise model m_in is lowest.
- P3 (readability, judged by eye, no test): the A_v maps of the real arm are concentrated inside the lesion around the seed, the flat arm's are weaker,
  the noise arm's near-empty; if the maps are as speckled as the raw gradients, the picture claim is dropped and only P1 / P2 are reported.
- Completeness relative error < 5 % (else M is raised).
