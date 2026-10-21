# LLD-MMRI-HCC — EVALUATION-ONLY dataset (cross-contrast HCC lesion segmentation)

> ⚠️ **This dataset is used for EVALUATION ONLY. No model is ever trained on it.**

LLD-MMRI-HCC is the **hepatocellular-carcinoma subset of LLD-MMRI** (Lou et al. 2025,
*Neural Networks*; MedSAM2-annotated release, Ma/Yang et al. 2025). Here it serves a
single purpose: **atlas-liver-hcc trains on a single modality (CE-T1w), so it cannot
answer the cross-contrast robustness question on its own** — this dataset supplies
**T2WI, DWI and Dixon in-/out-phase** (genuinely different contrast mechanisms from
CE-T1w) with real HCC lesion masks, letting atlas-liver-hcc-trained models be tested out-of-modality on
the same disease (HCC).

## Why no training here

- There is **no `01_create_splits`, `03_preprocess`, or `04_train`** stage — those
  `5_scripts_lld-mmri-hcc/` subdirs are intentionally empty (no `4_splits`/preprocessed
  content either).
- The pipeline only **consumes** atlas-liver-hcc-trained checkpoints: `05_predict` runs
  `nnUNetv2_predict` against the atlas-liver-hcc model folders, and `06_evaluate` scores
  the results against this dataset's own lesion ground truth.

## Data

- **157 HCC patients** (category `Hepatocellular_carcinoma` = class 6, filtered from
  the full 498-patient / 7-lesion-type LLD-MMRI dataset). Every patient is imaged in
  **all eight phases** the source provides, giving **eight test items** (same 157
  patients in each, so more items, not more patients). Items are built by
  `5_scripts_lld-mmri-hcc/05_predict/05_00_build_test_inputs.py` into
  `2_nnUNet_lld-mmri-hcc/raw/imagesTs_<item>/` + `labelsTs_<item>/`:

  | Item | Contrast | Role for the ATLAS CE-T1w-trained models |
  |---|---|---|
  | `ce-pre` | T1 pre-contrast | in-domain |
  | `ce-art`, `ce-pv`, `ce-del` | T1 contrast-enhanced (arterial, portal-venous, delayed) | in-domain |
  | `t2wi` | T2-weighted | held-out (OOD) |
  | `dwi` | diffusion-weighted | held-out (OOD) |
  | `inphase` | Dixon in-phase T1 | held-out (OOD) |
  | `outphase` | Dixon out-of-phase T1 | held-out (OOD) |

- **In-domain vs OOD.** The training set of this task is `../atlas-liver-hcc`, whose
  cohort mixes CE-T1w phases, so **every T1 phase (pre-contrast included) counts as
  in-domain** and is pooled into one group (`CE_T1w_all`) in
  `atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/configs/atlas-liver-hcc_combined_01_results.yaml`.
  The four contrasts T2WI, DWI, in-phase and out-phase are the held-out contrasts.
  The HCC causal-ablation ladder (`atlas-liver-hcc/.../06_evaluate/06_13_ladder_summary.py`)
  uses the grouped engine with OOD = these four items of this dataset only (the CE
  phases are excluded as same-contrast).
- Source: `wanglab/LLD-MMRI-MedSAM2` on HuggingFace (public, no gate; HCC patients only,
  not the full ~6 GB 7-class repo). Image and label files are hard-linked into
  `1_BIDS_lld-mmri-hcc/` by `00_utils/00_01_bidsify.py`.
- **No pre-processing before prediction**: no crop, no resampling, no reorientation
  (orientation was verified consistent at onboarding), and no FOV restriction.
- Segmentation is **lesion-only** (binary `{0,1}`) — annotated via MedSAM2 in a
  human-in-the-loop pipeline, **not fully expert-drawn from scratch** like
  atlas-liver-hcc's masks. Verified organic (non-box) shape on a sample case before
  onboarding, but this quality caveat should be kept in mind when interpreting results —
  a weaker cross-contrast Dice here could partly reflect annotation noise, not just
  domain gap.
- No liver-organ mask exists in this dataset — **only the `tumour` label is scored**,
  never `liver`.
- Subject ids are sequential and BIDS-legal; the mapping to the source ids is kept
  locally (`participants.tsv`, `case_id_map.json`) and is not part of the public repo.
- Metrics land in
  `8_results_lld-mmri-hcc/02_metrics/atlas_liver_hcc_model/t1w/<category>_<RUN_ID>/fold{k}/`
  (ladder rungs under `.../t1w/ablations/`). SRCSM rows use test-time source matching
  (the `_srcmatch` run directory).

## Evaluation specifics

- atlas-liver-hcc models emit 3 labels (0=background, 1=liver, 2=tumour). This
  dataset's GT is lesion-only, stored as **GT label 1**. Because atlas's tumour id (2)
  and this dataset's GT id (1) don't match numerically, evaluation uses the shared
  evaluator's **`--label_map '{"tumour": [2, 1]}'`** cross-label-space mode — the same
  mechanism `msd-spleen` uses for chaos's spleen id (4) vs its own GT id (1).
- No FOV restriction (unlike the chaos-family cross-eval sets) — a lesion target has
  no organ-boundary anchor concept.
- Single input channel per item matches atlas-liver-hcc's single-channel
  T1w model. nnU-Net applies atlas's T1w normalization statistics to genuinely
  different-contrast input — intentional; this is exactly what the domain-randomization
  methods are built to absorb.
- Methods evaluated: the atlas-liver-hcc 6-method suite (`baseline`, `auglab_default`,
  `synthseg_noEM`, `synthseg_EM`, `srcsm`, OURS
  `auglabAug_v26_6_2_train050_val000`/`val100`) — see `05_predict/05_0[2-8]_predict_atlas_*.sh`.
- Fold policy: **folds 0 1 2 only** (project-wide policy, 2026-07-09 onward).

## License note

LLD-MMRI is distributed under a custom data use agreement (despite the CC BY-NC badge on its
README): no copying, publishing or distribution of any part of the dataset, annotations and
cropped image parts included, outside internal use at a single site. Images, masks and case-ID
lists therefore never go into this repository, git-annex or public figures. Cite Lou et al.,
*Neural Networks* (2025) for LLD-MMRI and Ma, Yang et al. (arXiv:2504.03600) for the MedSAM2
segmentation annotation.
