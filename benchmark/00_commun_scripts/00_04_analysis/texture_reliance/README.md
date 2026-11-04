# Texture reliance: does scrambling fine texture change the texture-fill model more than the flat-fill model?

Preregistration, written and committed on 2026-11-03 before any prediction was made and before the perturbed test
sets were generated (only a 3-image pilot of the operator, on openly licensed MS images, was run to set the QC gates). Nothing in the design, the scores, the statistics or the verdict rule below changes after the first result.
Results are never committed (public repository); they are written to `/scratch/paulh/texture_reliance/` on Vulcan.

## Question and claim

The ablation ladders compare two networks that differ only in how the synthetic half of their training images fills
each region: rung 4.5 "flat fill" (region mean) and rung 5 "texture fill" (PA alone, real within-region intensities,
val000). This test asks, with an intervention at test time and no retraining:

> Does scrambling fine texture change the texture-fill model's predictions more than the flat-fill model's?

The claim the test can support is exactly that sentence, per training domain and across training domains. It says
nothing about other perturbation types and nothing about robustness in general; the report states only this.

## Models, domains, items (same as the paper's ladders)

- 9 ablation tasks, 19 training domains: Glioma x4 (t1n, t2w, t2f, t1c), MS x2 (flair, t1w), Breast x2 (t1wce, t2w),
  HCC x1 (t1w), Abdomen x2 (t1in, t2spir), Brain x3 (T1w, T2w, dwi_ap), Mandible x1 (cbct), Pelvis x2 (ct, mri),
  Pancreas x2 (t1wce, t2w). Groups (descriptive only): appearance = Glioma, MS, Breast, HCC; interface = Abdomen,
  Brain, Mandible, Pelvis, Pancreas.
- Per domain, the flat model is `run_keys[4]` and the texture model `run_keys[5]` of that domain's
  `ablations/ladder_series.json` (labels checked: "flat fill (rung 4.5)" / "... (real fill)"; flat runs registered in
  `benchmark/00_commun_scripts/00_03_evaluate/ladder_flat_rung.yaml`). Folds 0, 1, 2; `checkpoint_best.pth`; the
  trainer name stored in each checkpoint must match the run's trainer dir.
- Test items = every out-of-domain item of that ladder's OOD pool, read with the ladder engine's own loader
  (`ladder_ood_common._labelled_case_values`, own contrasts + extra cohorts, cross-dataset ladders included), and the
  same cases. Strata = the ladder's `ood_groups` (grouped ladders) or one stratum per item.
- Inputs = the exact directories the ladder's original predictions were made from (read from their
  `predict_from_raw_data_args.json`, remapped to the current Vulcan layout): crop-before-predict crops for amos /
  sliver07, `*_uniap` crops for duke / acrin6698, S-I flipped `*_sif` inputs for HaN-Seg / PDDCA, per-case FOV crops
  for the pancreas companions, RAS inputs for on-harmony. The Abdomen t2spir models were predicted from two copies of
  the same crops; `build_manifest.py` checks that the copies are identical (arrays and affines) and uses one.

## Perturbation T (label-free; no GT mask anywhere)

On each test input `x`, in its own voxel grid (as read by nnU-Net, SimpleITK):

1. `body` = foreground of `x` alone: min-max of `x` clipped to its own [0.5, 99.5] percentiles, `> 0.05`, binary
   closing (2 iterations), holes filled, connected components smaller than 1% of the mask dropped.
2. `low = G_2(x)` (Gaussian, sigma = 2 voxels on every axis), `band = x - low`: the fine + mid detail above the
   sigma-2 scale.
3. `scr` = phase-scrambled `band`: same Fourier amplitude spectrum, random Hermitian-consistent phases (the phases of
   the rfftn of real white noise; global DC bin kept); local energy restored by the gain
   `g = sqrt((G_2(band^2) + eps) / (G_2(scr^2) + eps))`, re-estimated and applied twice, `eps = 1e-3 mean(band^2)`
   in the body; masked to the body; rescaled so its body energy equals the band's. This is, verbatim, the
   "allfine_le" operator of FINDINGS 41 (`brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/
   scripts/visual_proof_scramble.py`), applied to whole test images instead of plan patches.
4. `x' = x - body * band + scr` (outside the body `x` is unchanged). Written as float32 with the input geometry.

One seed per case: `numpy.random.default_rng(crc32("<task>/<dataset>/<item>/<case>"))`, so one perturbed image per
case, shared by both models and by every training domain of the task that tests that input. No control perturbation
is run (decided before the experiment).

### QC of the operator (gates fixed now; values per image, inside the body)

- `scr_corr` = corr(band, scr): structure destroyed, gate `|scr_corr| <= 0.05`;
- `le_corr` = corr of the log local-energy maps `G_2(band^2)` and `G_2(scr^2)`: local energy restored, gate `>= 0.80`;
- `fine_corr` = corr(x - G_2 x, x' - G_2 x'): the fine band of the perturbed image no longer follows the original,
  gate `<= 0.50` (a perturbed image's re-extracted band keeps the `G_2(band)` part of the original by construction);
- `lp4_frac` = rms(G_4(x') - G_4(x)) / rms(x' - x): share of the change that survives a sigma-4 low-pass, gate `<= 0.25`;
- `lp4_rel` = rms(G_4(x') - G_4(x)) / std(x): the low-pass image is essentially unchanged, gate `<= 0.15`.

The operator passes QC when at least 95% of the perturbed images pass every gate. Positive control (the gates must
respond): 3 images per openly licensed input scrambled at every scale (`perturb.py --mode control_all`, band =
`x - mean`) must fail (at most 5% passing); an unperturbed image fails `scr_corr`/`fine_corr` by definition. QC slice
images are made only from openly licensed cohorts (open-ms, chaos, ispy2, totalseg-pelvic, on-harmony); never from
LLD-MMRI (DUA), ISLES, HaN-Seg (CC BY-NC-ND), PDDCA, BraTS or any other restricted cohort. A QC failure is reported,
not fixed silently; all cases stay in the primary analysis.

## Predictions

Both conditions (original and perturbed) are predicted fresh on Vulcan L40S in the same job, with the same model
object, so the agreement score contains no cross-machine or cross-software noise. The earlier texture-removal
experiment's re-predicted originals ("origts") are not reused: they cover 8 of the 19 domains and were made on
another machine. The call is the shared predict driver's (`nnUNetv2_predict -c 3d_fullres -f <k> --disable_tta
-chk checkpoint_best.pth`, nnU-Net defaults: tile step 0.5, Gaussian weighting, no mirroring) through nnU-Net's Python
entry point on the run's model folder. Brain (on-harmony) predictions of both conditions are resampled to the native
grid by the dataset's own post-item hook (`05_02_resample_predictions_to_native.py`), as the benchmark does.

## Scores

Per model, fold, item, case and evaluated label (the `label` column of the ladder's original `eval_all.csv`; label ids
from the model's `dataset.json`), with `eval_metrics.dice_score` (NaN when both masks are empty, then dropped):

- **Primary (no GT):** agreement `A = Dice(pred(x'), pred(x))`; case sensitivity `S = 1 - mean(A)` over the finite
  (label, fold) values, the same averaging rule as the ladder engine (`load_case_means`).
- **Secondary:** GT Dice drop `G = mean Dice(pred(x), GT) - mean Dice(pred(x'), GT)` (same averaging; GT and label
  maps as the dataset evaluators use them: amos organ id map, liver-only sliver07 / CHAOS CT, mandible-only HaN-Seg /
  PDDCA with the S-I flipped CT GT, BraTS labelsTr, on-harmony native GT). Checked by comparing `Dice(pred(x), GT)`
  of these re-predictions with the ladder's own `eval_all.csv` for the same model, fold, case and label
  (`<domain>_validation.tsv`).
- Domain value = equal weight per OOD stratum, cases pooled inside a stratum (the ladder's grouped estimand).
  **Effect per domain = S_texture - S_flat** (positive: the texture-fill model's predictions change more); GT effect =
  G_texture - G_flat.

## Statistics

- Within a domain: patient-level sign-flip test of the effect (`stat_tests.macro_perm_design`): units = physical
  patients via `ladder_ood_common._patient_key` (a patient in several items counted once; I-SPY2 `_uni`/`_bil` merged;
  on-harmony unit = session; totalseg-pelvic CT and MRI cohorts kept apart), one-sided (texture > flat) and two-sided;
  Holm over the 19 domains reported alongside.
- Across domains: exact sign test over the 19 training domains (`stat_tests.sign_test`, S_texture vs S_flat). No Dice
  or sensitivity is ever averaged across tasks; patients are never pooled across tasks.
- Appearance vs interface split: counts and medians only, descriptive, not a prediction.

## Prediction and verdict rule

Prediction: effect > 0 in most training domains, regardless of task group.

- **SUPPORTED** if the sign test over the 19 domains is significant (one-sided exact p < 0.05, i.e. >= 14/19 with
  S_texture > S_flat) AND at least 10/19 domains have effect > 0 with within-domain one-sided p < 0.05.
- **KILLED** if at most 9/19 domains have effect > 0 with p < 0.05 and the sign test is not significant (the effect is
  <= 0 or not significant in most training domains).
- **INCONCLUSIVE** otherwise. The secondary GT-Dice result is reported with the same tests and does not change the verdict.

## Run (Vulcan)

```
bash 01_build.sh       # manifest + checks (CPU job, blocking)
bash run_chain.sh      # 02 perturb -> 03 QC + 04 predict (GPU) -> 05 score -> 06 analyze, Slurm dependencies
```
Outputs (scratch): `manifest.json`, `images/`, `gen_stats/`, `qc_png/`, `pred/`, `score/`, `analysis/REPORT.md`,
`chain_jobs.txt`, `logs/`.
