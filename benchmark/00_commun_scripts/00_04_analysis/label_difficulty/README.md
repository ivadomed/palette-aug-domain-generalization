# Label-level difficulty vs target type: preregistration (2026-11-03)

Written and committed BEFORE any per-label number was computed. Only the label roster per setting (which labels
exist in `eval_all.csv`) was looked at.

## Question

The texture step (flat fill, rung 4.5 -> texture fill "PA alone", rung 5) helps on held-out contrasts for
appearance-defined targets and not for interface-defined ones. Rival explanation: appearance targets are harder,
and texture helps hard targets. Across tasks, difficulty and task type are nearly collinear. Inside one task the
models, data and contrasts are fixed while labels differ a lot in difficulty, so label-level variation can separate
the two explanations. No new training.

## Data

- Settings: every ladder setting whose task has >= 2 foreground labels in `eval_all.csv`.
  - Included (11): Glioma t1n / t2w / t2f / t1c (appearance; labels ET, NCR, RC, SNFH); Brain (ON-Harmony)
    T1w / T2w / dwi_ap (interface; 31 SynthSeg labels); Abdomen t1in / t2spir (interface; liver, left/right kidney,
    spleen); Pelvis ct / mri (interface; 11 labels).
  - Excluded, one foreground label only: MS (lesion), Breast (tumour), HCC (tumour), Pancreas (pancreas),
    Mandible (mandible).
- Runs and OOD pools: exactly those of each setting's own ladder script (the one listed in
  `scripts/cluster/rung5_val000/run_ladders.sh`; Brain dwi_ap = the default, not `--restricted`). The rung keys,
  in-domain contrast, OOD contrasts, extra sources and contrast groups are captured from the script itself (its
  `run_ladder*` call is intercepted), and every number is computed by the shared engine
  `ladder_ood_common.py` itself, with its CSV reader restricted to one label (or one label set). So per-label
  values use the engine's own fold cap, pooling (equal weight per held-out contrast / contrast group, cases pooled
  inside a group), patient merging (`_patient_key`) and paired Wilcoxon. Flat rung = `ladder_flat_rung.yaml`.

## Quantities per (setting, label)

- gain = texture-fill OOD Dice - flat-fill OOD Dice (percentage points), engine pooling.
- Difficulty (higher Dice = easier; "harder" = lower Dice):
  - (a) flat-fill OOD Dice of the label (requested primary; same models).
  - (c) noise-fill (rung 4) OOD Dice of the label: an independent training run on the same held-out pool.
  - (d) (texture + flat) / 2 OOD Dice (Oldham).
  - (b) plain nnU-Net baseline (ladder rung 0) in-domain Dice of the label (secondary).
- Covariate: median non-empty label volume (mL) over up to 20 `labelsTr` cases of the setting's training dataset.
- Within-setting per-label test: the engine's paired Wilcoxon on patient-merged pairs (flat vs texture).

Why (c) is decisive, not (a): gain = T - F regressed on F carries -Var(F) under the null, so test-set noise shared
between F and the gain, plus the Dice ceiling, produce a negative slope on (a) by construction. (c) and (b) share
no model with the gain; (d) removes the coupling under equal variances. (a) is reported but cannot by itself support
the difficulty explanation.

## Tests

- Per setting: Spearman rho(gain, difficulty) across labels, for (a), (b), (c), (d); permutation p (exact for
  n <= 8 labels, else 20000 permutations), two-sided. Partial Spearman controlling log volume (ranks residualised on
  the rank of log volume), only in settings with >= 8 labels (Brain, Pelvis).
- Across settings (unit = SETTING; no patient pooling across tasks, no Dice averaged across tasks): exact sign test
  on the sign of rho, over the 7 interface settings (primary) and over all 11 (secondary), for each difficulty
  measure.
- Hardest tercile: in Brain and Pelvis settings, the third of labels with the lowest (c) (ceil(n/3) labels; ties by
  label name), pooled as one label set through the engine; flat -> texture delta and engine Wilcoxon p.
- Glioma, descriptive: SNFH (edema) is pre-declared as the sub-region expected to carry the gain (earlier per-region
  analysis); report the per-region gain order.

## Power, stated up front

Spearman over 4 labels (Glioma, Abdomen) cannot reach p < 0.083 even with perfect order; only Brain (31 labels) and
Pelvis (11) carry within-setting inference, and all five are interface settings. Sign test minimum two-sided p:
0.016 over 7 settings, 0.001 over 11. The appearance-vs-interface comparison at label level is therefore descriptive
(slope signs + one figure), the within-interface test is the powered one.

## Predictions and kill criteria

- Difficulty explanation (H_D) predicts harder labels gain more INSIDE interface tasks.
- Target-type explanation (H_T) predicts no systematic slope inside interface tasks, and hard interface labels
  near zero gain.

Decision, on difficulty (c):
- H_D supported (target-type claim killed at label level) if ALL of: rho < 0 in >= 5 of 7 interface settings; rho < 0
  with p < 0.05 in >= 3 of the 5 powered settings (Brain x3, Pelvis x2); the hardest-tercile gain is > 0 with
  p < 0.05 in >= 3 of those 5.
- H_T supported if: rho < 0 in <= 4 of 7 interface settings OR p < 0.05 negative in <= 1 of the 5 powered settings,
  AND the hardest-tercile gain is not significantly > 0 in >= 4 of the 5.
- Anything else: mixed / inconclusive, reported as such.
- A negative slope on (a) alone, not seen on (c) or (d), counts as the coupling artefact, not as support for H_D.

## Sanity check (before reading any slope)

(i) The all-label (unrestricted) rerun of each captured ladder must reproduce that setting's on-disk
`ladder_series.json` flat -> real Dice step to rounding. (ii) The mean over labels of the per-label gains is
reported next to it; differences come from the engine being case-first (labels averaged inside a case, NaN labels
dropped) and are reported, with per-label NaN counts.

## Output

Code and this file are committed. Results (csv, md, png) go to `outputs/` (gitignored) or `$SCRATCH`, never to git
(public repo; no results or case ids in commits).
