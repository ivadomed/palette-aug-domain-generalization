# Changelog

This changelog summarises the code history of the repository by month, newest first. It is generated from the commit history (describing code and process changes only, not results). Each month lists its main themes followed by a collapsible list of every commit grouped by area. For the full one-line history run `git hist` (`scripts/git/hist.sh`).

## 2026-10

- Public-repo hygiene on 2026-10-09: internal notes, manuscript and dashboard untracked, LICENSE and CI added, README rewritten with a project overview (`6579c6a`, `6259ba2`); PR #30 and PR #31 merged (`2526545`).
- Roster-driven shared predict/evaluate drivers, a pipeline scaffolder and an eval-only companion toolchain were added; the isles2022 task was onboarded, then archived as a scaffolding reference (`32906ad`, `81c3f5f`).
- New pancreas task (pansegdata) with TotalSegmentator-MRI and MSD-CT eval companions; training moved between Killarney and Vulcan (`c1ff411`, `8268e53`, `ce1198b`).
- Ladder engine changes: rung 5 selected on val000, rung 4 retrained with label Voronoi, a flat-fill rung 4.5 inserted before real fill, rung-6 PV wrappers for all training sets (`e8f5ec6`, `e6c3dce`, `a8d0e4f`, `71bcb42`).
- ispy1 and acrin6698 added as breast companions, A-P crop standard, a fold-3 contamination fix in the ladder engine, SRCSM source matching for predict/eval (`1726366`, `665e5b5`, `17390ee`).
- A long series of texture-mechanism analysis scripts (visual-proof v7-v19, pre-registered texture-rule scripts, reduced-AugLab variants, latency probe) (`9babfae`, `43a5b4a`, `5bc4124`, `04967d2`).
- Git workflow adoption: commit/push checks, git hooks, health audit, `hist.sh` readable history, verbatim MIT LICENSE, CI lint fixes, and tracking of previously untracked pipeline scripts (`42b4916`, `993c164`, `c8584ce`).
- Paper-script and manuscript work through early October (task naming, figures, tables, review passes), later moved out of the public repo (`a88aef3`, `71bcb42`, `6579c6a`).

<details><summary>All 80 commits</summary>

**ladder**
- `665e5b5` fix fold3 contamination, add grouped OOD pooling
- `71bcb42` add rung-6 PV wrappers and rung-5 retrain launcher
- `25ad970` queue rung-6 PV post-training predict and eval
- `eef84d9` add PV branch engine hook and rung-6 rosters
- `035599a` add final PV rung wrappers, packs and queued jobs
- `e8f5ec6` select rung-5 checkpoint on real validation
- `3ecaf74` retrain rung 5 on val000 for five tasks
- `c143b8c` point ladders and PV registry at val000 rung 5
- `e6c3dce` retrain rung 4 with label_voronoi
- `73886f3` repoint ladders to lblvor rung 4
- `a2cac47` add rung 4.5 flat-fill wrappers and controllers
- `41d2fce` add items filter for extra OOD source
- `a8d0e4f` insert flat-fill rung 4.5 before real fill
- `a7fc893` add read-only ladder audit

**ispy1**
- `1726366` onboard breast eval companions, add A-P crop standard

**brats2024-glioma**
- `859a078` add radiomics scripts for edema analysis
- `f50c6e2` add rung-6 PV t1n wrapper and self-tests
- `bd45f07` add exploratory fill-swap rule tuning
- `76c99df` track visual-proof launchers and pilots

**paper-scripts**
- `d6a7ffb` fix post-restructure paths and cover all ladders

**shared**
- `a88aef3` run meta heatmap cross-task test on patient units
- `0acd03a` add per-task significance markers to meta heatmap
- `32906ad` add roster-driven predict/eval drivers and scaffolder
- `f8701fe` add eval-only companion toolchain
- `a2795d3` neutralize scaffolder epoch-policy TODO
- `17390ee` add SRCSM test-time source matching
- `45eeb7f` add ood_only mode to meta task heatmap
- `af7a303` guard per-dataset DA_WORKERS and nnUNet_compile
- `993c164` log code commits per training; add hist.sh

**results**
- `1499856` re-import spine results, regenerate BraTS supplement
- `d9cafbb` refresh ladder figures and tables for cross-dataset chaos
- `b480d09` use task names, add NGF violin and per-task tables
- `3bd60ef` use SRCSM source-matched runs in headline configs

**chaos**
- `ece9d17` make crop-before-predict standard, add cross-dataset ladders

**isles2022**
- `2add174` add brain_stroke training task
- `b00def0` finalize epochs, probe and launch training
- `c62f88d` add headline configs and record training results
- `81c3f5f` add ADC polarity control, archive task

**pansegdata**
- `c1ff411` onboard pancreas task and launch on Killarney
- `5621d1c` move training folds to Vulcan
- `3cf3581` generate results and significance configs
- `b9e5e6d` launch rung 4.5 flat-fill folds

**on-harmony**
- `239830a` resample RAS to native inside predict jobs
- `60f04fd` make checkpoint_best without TTA the default

**analysis**
- `f585bb1` refresh texture fill-swap study on val000 retrains
- `87a1e85` add surroundings texture-complexity test
- `73d993c` add train-me course for texture study
- `1d132f0` add visibility sign rule and example figures
- `a0afec7` flat-fill visibility sheets and fill-swap decomposition
- `9babfae` add visual-proof scale-space and v7 gradient analyses
- `594cc43` add visual-proof v8 evidence maps and probes
- `4860f52` add counterfactual, probe, CKA and gradient-spectra studies
- `24b90e1` add Fourier/wavelet and outcome-stratified evidence (v9)
- `b9c0945` add path-decomposed gradients and directional derivatives
- `3e51c5e` harden grain analysis
- `37e4f61` add per-fold grain and radial spectrum
- `7c62c4d` add texture-switch, stitching and scale-band studies
- `43a5b4a` add texture-evidence maps, integrated gradients, LEACE
- `5bc4124` add DCT kernel spectra and kernel smoothing
- `87267d9` add polarity and interface-sharpness texture rules
- `c05cb28` add edge-gap, shared-texture and difficulty rules

**totalsegmri-pancreas**
- `8268e53` add eval companion

**totalseg-pelvic**
- `46a1c46` add per-label view of the ladders

**repo**
- `0f4546e` archive retired val100 runs
- `df7a474` archive legacy eval configs and comparison tables
- `6579c6a` untrack internal notes and manuscript, add LICENSE and CI

**msd-pancreas**
- `ce1198b` add CT eval companion

**docs**
- `6259ba2` rewrite README with overview, video and results
- `2526545` fix the README video link (edit made on GitHub)

**auglab**
- `ecbdfad` add texlite, keepres and edgeonly reduced-AugLab arms

**cluster**
- `04967d2` add per-epoch latency probe
- `6830f66` add early-read tooling for reduced-AugLab arms
- `e192727` add PA variants round 2 wrappers and launcher

**git**
- `42b4916` add commit/push checks and hooks; fix CI lint and LICENSE
- `235b9ce` post-rewrite follow-ups: CHANGELOG, CI on main, hist.sh

**atlas-liver-hcc**
- `c8584ce` track the liver HCC pipeline and companion scripts

**benchmark**
- `aba4cbd` track the pancreas companion scripts

</details>

## 2026-09

- Restructure of `datasets/` into `benchmark/02_tasks/<task>/<dataset>` with follow-up path fixes (`6b4c4e6`, `e796fb4`).
- Canonical `06_evaluate` entry-point numbering applied across datasets; validator and dataset scaffolder updated (`1de4586`, `3ad8174`).
- New tasks onboarded: toothfairy2 with hanseg and pddca eval sets, I-SPY2 with duke-breast-mri, totalseg-pelvic; an autopet scaffold was added then removed (`43ac73c`, `35c9a42`, `21dccf2`, `46ba122`).
- Atlas-liver-hcc archived early in the month (`8549cf2`); a PET-task handoff doc was written (`411b742`).
- BraTS gained T2f and T1c training modalities, on-harmony gained dwi_ap, and significance testing switched to patient-level units (`d508f43`, `aac7f14`, `a968d21`, `34c80aa`).
- Breast unilateral-crop evaluation became the standard for duke-breast-mri; ladders pool same-patient pairs (`6dc8ab8`, `8650f92`).
- Cluster tooling: TamIA/Killarney relay helpers, pack GPU-map generator, tmux session notes (`59e4856`, `e11b83a`, `d508f43`).

<details><summary>All 44 commits</summary>

**repo**
- `8549cf2` archive atlas-liver-hcc extension
- `46ba122` remove abandoned autopet scaffold
- `fbfc149` stop tracking non-README markdown notes
- `6b4c4e6` restructure datasets/ into benchmark/02_tasks/<task>/<dataset>
- `e796fb4` fix broken path references after restructure
- `1de4586` enforce and scaffold canonical 06_evaluate layout

**ladder**
- `a6f84fe` make plots OOD-only
- `ceb9eea` add two new tasks to the ladder figure

**toothfairy2**
- `43ac73c` onboard maxillofacial CBCT task with hanseg test set
- `01d1273` add provenance, hanseg MRI arm and mandible table
- `e148a73` make mandible-only the sole standard

**hanseg**
- `411b742` add MR-T1 arm with propagated ground truth

**results**
- `de7be50` add breast and dental tasks to meta heatmap and ladders

**shared**
- `134d507` contrast_groups controls displayed columns
- `8dda82e` justify sign-flip macro-delta test
- `34c80aa` use patient as independence unit in significance
- `8650f92` align label pooling and pool same-patient ladder pairs
- `290b94d` harmonize 06_evaluate entry-point numbering
- `3ad8174` finish output prefix renames and path fixes

**atlas-liver-hcc**
- `39d87fb` extend lld-mmri-hcc phases and fix CE-T1w pooling

**analysis**
- `83bccb7` add domain-distance and cross-dataset ladder tools
- `c2db88f` add level-1 texture analysis for brats and ispy2

**cluster**
- `59e4856` add TamIA and Killarney relay helpers

**autopet**
- `eafde5b` onboard PET+CT training task

**ispy2**
- `35c9a42` onboard breast task training set and duke eval companion
- `f715ac4` add own-model significance and duke unilateral crop

**totalseg-pelvic**
- `21dccf2` onboard CT+MRI pelvic task
- `94222f0` wire real run ids and ladder methods
- `35e6f12` keep ablation rungs out of headline tables

**brats2024-glioma**
- `d508f43` add T2f/FLAIR training modality
- `6954d24` add TamIA pack launcher for T2f suite
- `98bc7ae` add T2f predict, eval and aggregation chain
- `aac7f14` add T1c as fourth training modality
- `2cb39fb` fix stale tamia_env hop after restructure

**pddca**
- `c178591` onboard second CT eval cohort for toothfairy2

**duke-breast-mri**
- `6dc8ab8` switch standard evaluation to unilateral crop

**on-harmony**
- `a968d21` onboard dwi_ap as third training modality

**benchmark**
- `8dbb47d` add hanseg node-pack launcher, register pelvis in heatmap
- `165766d` guard sig_ref and fix post-restructure doc issues
- `1efbdb8` standardize on-harmony, brats and spine script layouts

**amos**
- `7e7f743` add config-driven per-organ aggregate

**infra**
- `6b4276f` auto-discover nnU-Net trainer shims in venv setup
- `e11b83a` add TamIA pack GPU-map generator

**docs**
- `b6be9da` add README

</details>

## 2026-08

- More eval-only companions for chaos models (kidney-t2w, cirrmri-liver) with TamIA pack jobs, plus a FOV hard-crop driver (`4a49ce2`, `56f4720`, `8b8812c`).
- Picai-prostate added as a training task, then untracked when abandoned (`203f2bc`).
- Shared combined-modality summary and task-level meta heatmap added, plus a cross-cluster monitoring dashboard (`d286513`).
- Atlas-liver-hcc onboarded with lld-mmri-hcc and liverhccseg companions, then archived together with excluded evaluation sets (`56d6f5c`, `8aded3a`, `971a752`, `8549cf2`).
- Shared ladder engine and `contrast_groups` pooling introduced (`a9e1bc7`).
- Healthy-spine-tum onboarded; paper scripts reorganised and the manuscript reframed (`60a40bf`, `d286513`, `cac93df`).

<details><summary>All 12 commits</summary>

**kidney-t2w**
- `4a49ce2` onboard eval set and investigate geometry

**chaos**
- `8b8812c` add FOV hard-crop driver, exclude fragmented eval sets

**picai-prostate**
- `203f2bc` add training dataset, then untrack it

**shared**
- `d286513` add combined-modality summary and task-level heatmap
- `f929062` fix dirname-after-cd bug and tighten structure validator

**atlas-liver-hcc**
- `56d6f5c` onboard HCC liver tumour segmentation task
- `8aded3a` onboard lld-mmri-hcc and liverhccseg eval sets

**repo**
- `971a752` archive excluded eval datasets and rename BIDS leaf routing

**ladder**
- `a9e1bc7` add shared ladder engine and contrast_groups pooling
- `cac93df` add atlas-liver-hcc cross-dataset ladder

**healthy-spine-tum**
- `60a40bf` onboard spine segmentation dataset

**on-harmony**
- `8f12f14` remove dead first-generation scripts

</details>

## 2026-07

- On-harmony texture generation/metric scripts and 31-class histogram coverage analysis, followed by open-ms coverage (`7b92548`, `e278ba2`).
- Shared layer consolidated: stat tests, run-all training launcher, `aggregate_from_config` moved into the shared evaluate layer, train/predict/aggregate drivers refactored (`019dd2e`, `3a2097b`, `7131b59`).
- Open-ms FLAIR suite and causal-ablation arms (noisefill, k-means, label remap, affine remap), with a Killarney handoff doc (`e278ba2`, `3a2097b`, `7634a67`, `5aad713`).
- Chaos T1in ablation ladder and the DualVal trainer ported to chaos and open-ms (`5aad713`).
- First eval-only companions onboarded: mslesseg, brats-ssa2024, ms3seg, msd-spleen, cirrmri-liver (`ecd1f5a`, `56f4720`).
- Manuscript revision commits and repo hygiene (dataset skeleton placeholders, unused datasets removed) (`cf8a10f`, `982107c`).

<details><summary>All 14 commits</summary>

**shared**
- `e91c12e` add common scripts, open-ms and FOV utilities
- `019dd2e` add stat tests and config-driven significance
- `7131b59` move aggregation to shared layer and refactor drivers

**analysis**
- `7b92548` add on-harmony texture metrics and histogram coverage

**open-ms**
- `e278ba2` add noisefill training and coverage analysis scripts
- `3a2097b` add noisefill-alone ablation arms and run-all launcher
- `212de1a` rename FLAIR ablation arms, add chaos eval configs

**benchmark**
- `d06fd46` add val000 train and predict scripts
- `ecd1f5a` onboard mslesseg, brats-ssa2024 and ms3seg eval sets

**ladder**
- `7634a67` add FLAIR val000 and kmeans ablation arms

**chaos**
- `5aad713` add T1in ablation ladder with DualVal trainer
- `56f4720` onboard msd-spleen and cirrmri-liver eval sets

**amos**
- `cf8a10f` add chaos SRCSM and val000 cross-eval predict scripts

**repo**
- `982107c` seed dataset skeleton and remove unused datasets

</details>

## 2026-06

- Project structure reorganised; nnU-Net trainer shim package added for brats2024-glioma (`a61dd5a`).
- New datasets and scripts added; scripts made flexible for Vulcan (`8696d57`).
- Hardcoded dataset names in amos predict scripts fixed (`8696d57`).

<details><summary>All 2 commits</summary>

**repo**
- `a61dd5a` reorganize project into dataset-centric layout

**infra**
- `8696d57` add new datasets and make scripts portable to Vulcan

</details>

## 2026-04

- Synthesis v18 model configs and variants, with logging updates (`14bcd08`).
- First nnU-Net implementation added (pre-cleanup) (`1dc2988`).

<details><summary>All 2 commits</summary>

**synthesis**
- `14bcd08` add BigAug baseline and v17/v18 model variants

**infra**
- `1dc2988` add first nnU-Net implementation

</details>

## 2026-03

- Initial project structure committed (`b3d6d43`).
- Segmenter training/evaluation updates (v4) with train/test leakage prevention (`b3d6d43`).
- Lightning and Kornia implementation, followed by repeated training-loop and augmentation speed-ups (`185c25b`).
- Synthesis fix for v6, BigAug baseline augmentations, v17 model configs and a refactor of versions and evaluation script (`185c25b`, `14bcd08`).
- Architecture, performance and knowledge-transfer docs written (`82d7797`).

<details><summary>All 3 commits</summary>

**synthesis**
- `b3d6d43` add initial generator, segmenter and evaluation scripts
- `185c25b` add Lightning/Kornia pipeline and speed up training

**docs**
- `82d7797` add architecture, performance and run-script docs

</details>
