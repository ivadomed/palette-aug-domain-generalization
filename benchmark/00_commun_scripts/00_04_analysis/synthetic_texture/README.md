# Synthetic texture experiment (preregistration)

Controlled, non-medical test of the benchmark's flat-fill vs texture-fill finding: with
randomized region contrast, keeping the real texture inside each region (PA texture fill)
instead of filling each region with its new mean (flat fill) should raise out-of-domain (OOD)
Dice when the target is defined by its texture, and do nothing when it is defined by a
boundary / mean step. Here the only thing varied is what defines the target.

This README was committed before any flat-fill or texture-fill result existed. Changes after
that commit are listed under "Calibration log" and "Deviations found while running".

## 1. Generator (`generator.py`)

2D images, 128 x 128, values in [0.03, 1.5] inside an elliptical body, 0 outside.

- **Anatomy.** The body is split into 4 smooth "tissue" regions (argmax of 4 Gaussian random
  fields smoothed with sigma = 9 px). 1-2 star-shaped targets (radius 7-14 px, harmonic
  perturbations) are placed in the body, each inside a 6 px ring forced to tissue 0, the
  **host**: the host is the target's only neighbour.
- **Tissue properties.** Tissue k has a base level b_k (0.2, 0.4, 0.6, 0.8) and a unit-variance
  band-pass Gaussian texture field with its own spectrum (centre frequency 0.05-0.15 cycles/px,
  some oriented). The target has its own spectrum L (0.20 cycles/px, oriented, orientation random
  per image). Tissue 3's texture is deliberately close to L, so "high-frequency oriented texture"
  alone does not identify the target.
- **Sweep, lam in {0, 0.2, 0.4, 0.6, 0.8, 1.0} (texture fraction).**
  target base level = b_host + (1 - lam) M, target texture = sqrt(1 - lam) H + sqrt(lam) L,
  where H is the host's own texture realisation (no seam at lam = 0) and L an independent field
  with the target spectrum. Both are unit variance, so the target's texture variance equals the
  host's at every lam. lam = 0: the target is the host's texture with a mean step (a boundary
  target, same texture as its neighbour). lam = 1: same mean and same variance as the host, the
  target differs only by the spatial pattern of its texture (no mean step, no intensity edge).
- **Contrast domains.** A domain is an intensity -> intensity function g_d, piecewise linear
  through knots at the base levels (knot values uniform in [0.15, 1.0], so non-monotone and
  inverting), with the knot at b_host + M set to g_d(b_host) +/- U[0.15, 0.30] (random sign).
  The target's level is interpolated on g_d, and its texture gain is the host's gain. Hence:
  at lam = 1 no domain can give the target a mean or variance cue; at lam = 0 every domain keeps
  a 0.15-0.30 mean step between target and host (polarity random). Per base level a signed
  texture gain in +/-[0.6, 1.5] (negative with p = 0.3: texture contrast inversion), acquisition
  noise sd in [0.005, 0.015], smooth multiplicative bias field up to +/-10%.
  Base texture sd 0.05 before gain.
- **Seeds.** training domain of seed s = `sample_domain(rng(1000 + s))`; 8 held-out test
  domains = `sample_domain(rng(9001 + i))`, i = 0..7, the same for every seed; training anatomies
  rng(20000 + s) (300 images), in-domain validation anatomies rng(30000 + s) (50 images), test
  anatomies rng(40000) (100 images, each rendered in all 8 test domains). Within a seed the
  anatomies, noise realisations and domains are the same for every lam, so lam is the only thing
  that changes between cells of a seed.

## 2. The ablation mirrored (`pa_arms.py`)

The fills are the AugLab classes themselves (no reimplementation), imported from a pinned copy
of the AugLab repo at commit f018536553ee096fb61d1bf566d70e89c5cec144 (`git archive` of
Vulcan's checkout, at `$SCRATCH/synthetic_texture/auglab_pin`, commit recorded in every run JSON).
The images are depth-1 volumes [B, 1, 1, H, W].

| arm | transform | paper counterpart |
|---|---|---|
| plain | none | baseline (no contrast transform) |
| flat | `RandomV26_6_2NoiseFillContrastGPU(noise_std_range=[0,0], label_fill_noise=False, label_voronoi=False)` | rung 4.5 flat fill, `..._flatfill_spatialDA_train050.json` |
| texture | `RandomV26_6_2ContrastGPU` | rung 5 "PA alone", `transform_params_gpu_v26_6_2_synth_spatialDA_train050.json` |

Kwargs are those of the two configs (which differ only in the class and the three noise-fill
kwargs): c_choices 2-6, s_choices 2-10, skip_parcellation 0.1, skip_sub_parc 0.4,
|alpha| in [0.5, 2], label_remap_prob 0.5, per-sample synthetic probability 0.5 (train050).
Same partition (k-means intensity clusters + Voronoi sub-parcellation), same new region means;
flat fill writes mu_c, texture fill writes mu_c + alpha_c (x - mean_c). The per-label remap
(step 2) runs in both arms on the target label. Every input (real or synthetic, train or test)
is foreground z-scored, as the transform's own output is.

## 3. Why the result is not built into the generator

- **The flat arm has the cue.** All arms see the real training images in 50% of samples
  (plain sees them always), so every arm can learn the texture cue from real images. The
  question is only whether the synthetic half generalises it across domains.
- **The flat arm can learn the target's shape and position.** The per-label remap gives the
  target its own random mean in ~50% of synthetic samples in both arms, so flat-fill images show
  the exact target shape as a region with its own intensity; k-means / Voronoi regions also
  follow intensity edges where they exist (lam near 0). Flat fill removes within-region texture,
  not the target.
- **The domains are fair to both ends.** The OOD shift is the same family at every lam (an
  intensity function + texture gain/sign), applied identically to target and host at lam = 1, so
  the generator never hands the texture end a new cue OOD, and never removes the boundary end's
  mean step (|step| >= 0.15 in every domain).
- **Not guaranteed either way.** Texture fill also randomizes texture gain and sign of every
  tissue, which could help (or hurt) at lam = 0 too; flat fill removes texture distractors
  (tissue 3 looks like the target), which could help it at lam = 1. The prediction can fail.

## 4. Training and evaluation (`train.py`)

2D U-Net (16-32-64-128 channels, 2 convs per level, instance norm, leaky ReLU, ~0.5 M
parameters), CE + soft Dice, AdamW lr 1e-3 with poly decay (0.9), batch 32, **4000 iterations for
every arm** (same budget), random flips + 90 degree rotations for every arm, final weights (no
checkpoint selection). Within a seed, the three arms share initialisation, batch order and spatial
augmentation (separate numpy RNG), so they are paired. Dice per image (target vs rest), in-domain
on the 50 validation images, OOD on 100 images x 8 held-out domains.

Grid: 6 lam x 8 seeds x 3 arms = 144 runs (`run_cells.sh`, one Slurm array task per (lam, seed),
three arms concurrently on one L40S).

## 5. Prediction, test and decision rule (fixed before any result)

**Prediction.** The OOD gain (texture - flat) grows as the target becomes texture-defined; it is
~0 for the boundary-defined end (lam = 0) and positive for the texture-defined end (lam = 1).

**Statistic.** Per run, OOD Dice = equal-weight mean over the 8 held-out domains of the mean
per-image Dice (points, x100). Per (seed, lam), gain = texture - flat. Seed is the replicate (n = 8).

- **T1 trend:** per seed, OLS slope of gain on lam (6 points); mean slope > 0 with an exact
  one-sided sign-flip test over seeds, p < 0.05.
- **T2 texture end:** mean gain at lam = 1 > 0, exact one-sided sign-flip over seeds, p < 0.05.
- **T3 boundary end:** |mean gain at lam = 0| <= 2 Dice points (margin fixed here; the benchmark's
  interface-bounded tasks were within +/-0.7).

**Verdict.** REPRODUCED if T1, T2 and T3 all hold; PARTIAL if exactly two hold; NOT REPRODUCED
otherwise (with the direction stated if a gain is significantly negative, two-sided sign-flip).

**Validity checks (reported, not part of the verdict):** V1 plain in-domain Dice >= 80 at every
lam (the task is learnable at both ends); V2 plain OOD Dice <= 0.8 x its in-domain Dice at lam = 0
and lam = 1 (the domains actually shift both ends). If V2 fails at one end, the conclusion for that
end is reported as uninformative.

**Reported but secondary:** per-lam texture - plain and flat - plain, 95% t-interval of the gain,
per-image Wilcoxon of the gain averaged over seeds (images are not independent replicates of
training, so this is not used for the verdict).

## 6. Calibration rule

Before the main launch, one pilot may adjust generator constants (texture amplitude, mean step,
domain ranges, iterations) using ONLY the plain arm (in-domain and OOD Dice at lam = 0 and 1,
seed 0) and the QC figure, to satisfy V1/V2. A timing probe runs the flat and texture arms for a
few hundred iterations without evaluation. Every change is listed below and committed before
the main launch.

### Calibration log

(none yet)

## 7. Deviations from the benchmark

- 2D (depth-1 volumes) instead of 3D; tiny U-Net instead of nnU-Net; fixed iterations and final
  weights instead of epochs + checkpoint_best on real validation.
- `blur_sigmas = [0.0]` in both arms (paper: [0, 0, 0, 0.3, 0.5, 0.8]): AugLab's separable 3D
  blur zero-pads the depth axis, so on a depth-1 image it scales intensities by the kernel centre
  weight before the label step, which the 3D runs never had.
- No AugLab intensity/spatial augmentations other than flips/rotations (the paper's PA-alone rung
  has none of AugLab's other transforms either, but it has nnU-Net's spatial augmentation).
- One target class (lesion-only style label map), so the per-label remap acts on the target only.

## Files

`generator.py` (images, domains), `pa_arms.py` (fills, from AugLab), `train.py` (one run),
`run_cells.sh` (Slurm array), `qc_figure.py` (example images, both fills), `analyze.py`
(preregistered tests). Outputs live on `$SCRATCH/synthetic_texture/`; the small final results are
copied to `/project/6102268/paulh/synthetic_texture_results_<date>/` (Killarney project space).
