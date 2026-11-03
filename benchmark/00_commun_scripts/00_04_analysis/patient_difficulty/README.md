# Patient-level, difficulty-matched test of difficulty vs target type: preregistration (2026-11-03)

Written and committed BEFORE any patient-level gain or any gain-vs-difficulty statistic was computed or looked at.
What had been seen at this point: (1) each setting's full-pool flat -> texture ladder step (already on disk in every
`ladder_series.json` and in the paper), reproduced by the sanity check below; (2) the per-patient DIFFICULTY
distributions (noise-fill and baseline OOD Dice quantiles per setting, and patient counts per candidate difficulty
window), used to fix the band rule. The per-setting JSON files holding patient gains were written by `compute` but not
opened.

## Question

The texture step (flat fill, rung 4.5 -> texture fill = "PA alone", rung 5, val000) helps on held-out contrasts for
appearance-defined targets and not for interface-defined ones. Rival explanation (H_D): appearance targets are just
harder, and texture helps hard cases. The label-level analysis (`../label_difficulty/`) found no difficulty slope across
labels inside interface tasks, but only Glioma represented appearance tasks there. Here the within-task unit is the
PATIENT, which works for single-label tasks too, so all 19 ladder settings / 9 tasks enter. No new training.

## Settings and types (19 settings, 9 tasks; = `ladder_flat_rung.yaml` and the paper's `compute_dissociation_pvalues.py`)

- Appearance-defined ("no tissue interface"), 4 tasks / 9 settings: Glioma t1n / t2w / t2f / t1c; MS flair / t1w;
  Breast t1wce / t2w; HCC t1w.
- Interface-bounded, 5 tasks / 10 settings: Brain (ON-Harmony) T1w / T2w (the paper's "FLAIR") / dwi_ap; Abdomen
  t1in / t2spir; Pelvis ct / mri; Mandible cbct; Pancreas t1wce / t2w.
- Ladder scripts: the ones each task's ladder runner uses (`run.sh` table; Brain dwi_ap = default full pool, not
  `--restricted`; MS flair = `06_19_ladder_summary_ood.py`; Mandible = `run_ladder_cross_dataset` with its
  contrast_groups, in-domain group removed).

## Per-patient quantities

The setting's ladder call is intercepted, the engine inserts the flat rung itself (`_with_flat_rung`), and per-case
OOD Dice (mean over labels and EVAL folds, `load_case_means`) is read for four rungs: baseline (rung 0), noise fill
(rung 4, lblvor), flat fill (4.5), texture fill (5). Pooling = exactly the engine's OOD figure: equal weight per OOD
contrast / extra item (plain `run_ladder`), per contrast group with member items' cases pooled (`ood_groups`), or the
contrast_groups tree (cross-dataset). Each case c on the flat/texture pair gets its ladder weight
w_c = (unit weight) / (cases in unit). Patient = the engine's patient key (`_patient_key`, same contrast argument as
the engine's significance code; I-SPY2 `_uni/_bil` merged; Pelvis CT/MRI cohorts kept apart).

- W_p = sum of w_c over the patient's cases (across all its contrasts / groups).
- gain_p = sum w_c (T_c - F_c) / W_p (texture - flat OOD Dice, pp).
- difficulty_p (PRIMARY) = sum w_c N_c / W_p, N = noise-fill OOD Dice (a separately trained model: no measurement
  noise shared with the gain). Higher = easier.
- baseline_p (SECONDARY difficulty) = same with plain nnU-Net (rung 0) OOD Dice.
- flat_p (COUPLED reference only): gain regressed on F carries -Var(F) under the null; reported, never decisive.
- So sum_p W_p gain_p / sum_p W_p equals the ladder's flat -> texture step exactly (sanity check). A patient seen in
  several contrast groups has ONE difficulty and ONE gain, both weighted the same way.

## A. Within-setting slope

Per setting: Spearman rho(gain_p, difficulty_p) over patients (unweighted), two-sided permutation p (exact for
n <= 8, else 20000 permutations, seed 0). Also for baseline_p and flat_p. Supplementary, pre-declared artefact checks:
(i) rho restricted to patients with noise Dice in [10, 85] (ceiling: a patient at 92 cannot gain 10; floor: a patient
at ~0 under every model gains nothing); (ii) within-unit rho (ranks of gain and difficulty centred within the
patient's unit) for settings where every patient belongs to exactly one unit.

H_D predicts rho < 0 inside interface settings. Sign test over the 10 interface settings (unit = setting, exact
binomial, two-sided).
- A supports H_D if rho < 0 in >= 8/10 interface settings AND rho < 0 with p < 0.05 in >= 3 of them AND rho < 0 in
  >= 7/10 in the [10, 85] restriction (a slope that exists only through the ceiling counts as artefact).
- A contradicts H_D if rho < 0 in <= 6/10 interface settings OR p < 0.05 negative in <= 1.
- Otherwise mixed. Appearance settings' rho are reported alongside, descriptively.

## B. Difficulty-matched comparison (main test)

Band rule (fixed from the difficulty distributions only): windows [a, b] of patient noise-fill Dice on a 5-point grid,
b - a >= 20. A setting QUALIFIES in a window if >= K = 5 of its patients fall in it; a task qualifies if >= 1 of its
settings does. Matching criterion: |median over qualifying appearance settings of their in-band median difficulty -
the same over interface settings| <= 5 pp. Among windows meeting it with >= 3 tasks of each type, take the most
qualifying tasks, then the most qualifying settings, then the smallest gap.
- Result of the rule: **PRIMARY band B1 = [55, 75]**: 8 tasks (appearance: Glioma (t1n, t2w), Breast (both), HCC;
  interface: all 5), 15 settings, difficulty gap -4.8 pp. MS drops out (every MS patient has noise Dice < 55).
- **Sensitivity band B2 = [25, 60]** (the best low-difficulty window under the same rule, gap -5.0 pp): 8 tasks
  (appearance: Glioma x4, MS t1w, Breast x2, HCC; interface: Abdomen x2, Mandible, Pancreas x2, Pelvis ct), 14
  settings. Brain drops (all Brain patients >= 52 with too few below 60).
- Power, stated up front: no window holds both MS (flair all <= 28.5) and Brain (all 52-73) with matched difficulty,
  so the band test runs over 8 of 9 tasks: C(8, 3) = 56 relabelings in B1 (min one-sided p = 0.018), C(8, 4) = 70 in
  B2 (min 0.014). Brain and MS have 8 patients per setting.

Per qualifying setting: band-gain = sum_{p in band} W_p gain_p / sum_{p in band} W_p (the ladder's case weights,
restricted to band patients); within-setting p = weighted sign-flip test of the in-band patient gains (statistic
sum W_p gain_p s_p, two-sided, exact for n <= 16 else 20000 flips).

Type comparison (unit = setting or task, never patients pooled across tasks, no Dice averaged across tasks):
(i) per type, exact two-sided sign test of the number of qualifying settings with band-gain > 0;
(ii) separation with TASK as the permutation unit: task value = mean of its qualifying settings' band-gains (within
task only); statistic = rank sum of the appearance tasks among the qualifying task values; exact p over every
relabeling of which qualifying tasks are "appearance" (keeping the type counts); one-sided (appearance > interface,
H_T's direction) is primary, two-sided reported.

Decision on B1 (B2 reported as sensitivity with the same rule):
- **H_T supported** if separation one-sided p <= 0.05 AND band-gain > 0 in all but at most one qualifying
  appearance setting AND the interface settings' band-gains are not consistently positive (interface sign test
  two-sided p > 0.05, or at most half of qualifying interface settings positive).
- **H_D supported** (target-type claim killed at matched difficulty) if separation one-sided p > 0.20 AND band-gain > 0
  in >= 2/3 of qualifying interface settings AND the median interface task value >= 0.5 x the median appearance task
  value (with the latter > 0).
- Otherwise **mixed / inconclusive**.

Secondary matching (reweighting): inside B1, four fixed 5-point bins (55-60, 60-65, 65-70, 70-75); a setting's
reweighted gain = equal-weight mean over bins of the W-weighted mean gain in each bin, computed only for settings with
>= 1 patient in every bin (a uniform common difficulty histogram; pooled deciles are not used because Breast would
define them). Same type comparison, run only if >= 3 tasks of each type qualify, otherwise reported descriptively.
Expected in advance: several interface settings (Brain, Pelvis) may fail the coverage requirement.

## C. Reporting

Per setting: n patients, n in each band, which settings fail K (and so leave the band test), band-gains with p.
Cross-task statements are over settings/tasks only. Caveats to report regardless of outcome: patient difficulty
partly reflects the contrast group the patient sits in (hence the within-unit rho); tasks with few OOD patients (Brain,
MS: 8 each); band patients of a task are a selected, possibly atypical subset.

## Sanity check (must pass before any result is read)

Per setting: sum_p W_p gain_p / sum W_p == on-disk `ladder_series.json` flat -> texture Dice step (to 1e-6 pp);
noise and baseline rung means rebuilt from the same units == on-disk values; patient count == the engine's
fill-swap `n_cases`. Passed for all 19 settings when this file was written (compute stage only).

## Output

Code and this file are committed. Results (json, csv, md, png) go to `outputs/` (gitignored), never to git (public
repo; no results or case ids in commits).
