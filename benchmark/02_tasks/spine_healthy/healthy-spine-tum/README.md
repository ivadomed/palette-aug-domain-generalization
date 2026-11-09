# healthy-spine-TUM — externally-computed results

> **Third drop, 2026-11-09** (same importer, `--verify` now checks every folder against the
> metrics already on disk). The six arms already imported reproduce exactly. New arms:
> - `SRCSM` → `srcsm` metrics `<run>_srcmatch` (SRC + test-time source matching, the reported
>   SRCSM, as for every other task); `SRCSM_no-source-matching` → the plain `srcsm` run next to it.
>   The headline `srcsm` row now points at the `_srcmatch` run.
> - `SynthsegEM05+GE` → `synthsegEM_train050_val000` (SynthSeg-EM at a 50% synthesis rate, the
>   "sem50" arm of the other tasks).
> - `ImageContrastV26-05+GE` reproduces `palette05_ge`'s Dice and adds HD95. It is Palette without
>   Auglab ("Palette" / PA in the paper); Spine has no ablation ladder, so the paper scripts take
>   it as the task's PA run directly.

> **Re-imported 2026-10-02** from the collaborators' second drop, which adds HD95.
> The importer is now `5_scripts_healthy-spine-tum/06_evaluate/06_00_import_tum_tsv.py`
> (replaces the one-off `_import_tum_tsv_20260805.py`, kept for provenance only in
> `benchmark/03_archive/healthy-spine-tum_oneoff_import_20260805/`).
> Two things changed besides HD95, both verified row-by-row:
> - Folder names: Ours is `ImageContrastV26-05+Paper` and Auglab is `newd` in the
>   new drop (each reproduces the old folder's Dice exactly on all 4536 rows, and
>   adds HD95). The script's `--verify` re-checks this for any future drop.
> - `synthseg_noEM` in the 2026-08-05 import was **truncated** (8-81 subjects per
>   test group instead of 27-219); now complete. Its combined Dice 50.4 -> 47.5.
> The superseded metrics are archived at `benchmark/03_archive/healthy-spine-tum_import_20260805/`.
> No SRCSM arm in this drop (added 2026-11-09, see above).


> ⚠️ **This dataset currently exists as `8_results_healthy-spine-tum/` ONLY.** No raw data,
> BIDS tree, nnUNet conversion, checkpoints, or split files live in this repo for this dataset —
> the metrics were computed on an external pipeline (not `run_job`/`00_commun_scripts`) and
> handed over as `tsv.zip`. `benchmark/validate_standard_dataset_structure.py` will correctly flag
> this dataset as missing slots 0–7 and 9 until/unless that data is onboarded here too.

## What this is

A 6-method comparison, trained on two contrasts (**CT** and **Dixon in-phase MRI**), evaluated
cross-contrast on two external spine test sets — **spider** (T1w, T2w) and **spinegan** (ct,
dixon-fat, dixon-inphase, dixon-water) — 3 folds each, mirroring this project's usual
train-one-modality / evaluate-cross-contrast methodology (see root the project notes).

## Method-name mapping (source dir → this project's roster)

| Source folder | This project's method | category | notes |
|---|---|---|---|
| `Base` | `baseline` | nnUNet | |
| `Paper` | `auglab_default` | auglab | |
| `Synthseg10NoEM+GE` | `synthseg_noEM` | auglab | |
| `Synthseg10+GE` | `synthseg_EM` | auglab | |
| `PALETTE-05+Paper` | `auglabAug_v26_6_2` (**OURS**) | auglab | user-confirmed 2026-08-05 |
| `PALETTE-05+GE` / `ImageContrastV26-05+GE` | `palette05_ge` | auglab | Palette without Auglab (PA); not a row of the fixed 7-method configs |
| `SRCSM` | `srcsm` (`..._srcmatch`) | auglab | SRC + source matching = reported SRCSM (2026-11-09) |
| `SRCSM_no-source-matching` | `srcsm` (plain) | auglab | 2026-11-09 |
| `SynthsegEM05+GE` | `synthsegEM_train050_val000` | auglab | SynthSeg-EM at 50% (2026-11-09) |

`palette05_ge` (PA) and `synthsegEM_train050_val000` (sem50) are not rows of the dataset's own
`06_evaluate/configs/*.yaml` tables, like PA and sem50 of every other task: the paper scripts add
them per setting (PA by run id, since Spine has no ladder).

Until 2026-11-09 there was no SRCSM arm and the `srcsm` row rendered blank.

## Train-contrast mapping

`SG_CT` → `ct`, `SG_in-phase` → `inphase`.

## Metric conversion (source TSV → our `eval_all.csv`)

Source TSVs are panoptic-quality-style per-structure metrics (`vertebra`/`ivd`/`spinal_canal`),
not plain Dice/HD95. Per user decision (2026-08-05):
- `dice` = `<label>-global_bin_dsc` (whole-structure binary Dice — the closest analog to how
  "dice" is used elsewhere in this project; NOT the same as `sq_dsc`/`pq_dsc`, which are
  instance-matching-sensitive).
- `hd95` = `<label>-global_bin_hd95` since the 2026-10-02 drop; a non-finite/blank value is
  written as `nan` (same as `evaluate.py` for an empty mask).
- `group` in `eval_all.csv` = `<test_set>_<test_contrast>`, e.g. `spider_T1w`, `spinegan_dixon_fat`.

Conversion script (one-off, not part of the repo's script layers — see the project notes' "put new
code in the right one" rule; kept here for provenance only, not intended to be re-run as-is):
`benchmark/03_archive/healthy-spine-tum_oneoff_import_20260805/_import_tum_tsv_20260805.py`,
superseded by `5_scripts_healthy-spine-tum/06_evaluate/06_00_import_tum_tsv.py`.

## Layout produced

```
8_results_healthy-spine-tum/
  01_predictions/   (empty — no prediction volumes in this drop, metrics only)
  02_metrics/healthy_spine_tum_model/{ct,inphase}/<category>_healthy-spine-tum_<contrast>_<method>_20260805_130846/fold{0,1,2}/eval_all.csv
```

Timestamp `20260805_130846` = import time (this repo never trained/predicted this data — there is
no real training/prediction timestamp to use).
