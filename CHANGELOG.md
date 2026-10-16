# Changelog

This changelog summarises the code history of the repository by month, newest first. It is generated from commit subjects (describing code and process changes only, not results). Each month lists its main themes followed by a collapsible list of every commit grouped by area. For the full one-line history run `git hist` (`scripts/git/hist.sh`).

## 2026-10

- Public-repo hygiene on 2026-10-09: internal notes, manuscript and dashboard untracked, LICENSE and CI added, README rewritten with a project overview (`4a23634`, `e732dbf`, `d2f490a`); PR #30 and PR #31 merged (`158cc9b`, `1ff80eb`).
- Roster-driven shared predict/evaluate drivers, a pipeline scaffolder and an eval-only companion toolchain were added; the isles2022 task was onboarded, then archived as a scaffolding reference (`b55c632`, `77c2751`, `8d89954`).
- New pancreas task (pansegdata) with TotalSegmentator-MRI and MSD-CT eval companions; training moved between Killarney and Vulcan (`49bf0e9`, `ce444bd`, `407ea24`).
- Ladder engine changes: rung 5 selected on val000, rung 4 retrained with label Voronoi, a flat-fill rung 4.5 inserted before real fill, rung-6 PV wrappers for all training sets (`99b03b9`, `a4f246d`, `23a9f80`, `7c9d12f`).
- ispy1 and acrin6698 added as breast companions, A-P crop standard, a fold-3 contamination fix in the ladder engine, SRCSM source matching for predict/eval (`d0a0f62`, `fbd5ff3`, `bd16c2b`).
- A long series of texture-mechanism analysis scripts (visual-proof v7-v19, pre-registered texture-rule scripts, reduced-AugLab variants, latency probe) (`d821e83`, `a66170d`, `c410eb8`, `54f12ca`).
- Git workflow adoption: commit/push checks, git hooks, health audit, `hist.sh` readable history, verbatim MIT LICENSE, CI lint fixes, and tracking of previously untracked pipeline scripts (`e2515c3`, `5ea7fe5`, `1b9d5e5`, `5a95e99`).
- Paper-script and manuscript work through early October (task naming, figures, tables, review passes), later moved out of the public repo (`837c329`, `42ab6ee`, `e732dbf`).

<details><summary>All 272 commits</summary>

**Merge pull request #30 from ivadomed/paul/v26_6_2**

- 158cc9b 

**Merge pull request #31 from ivadomed/paul/v26_6_2**

- 1ff80eb 

**amos-pancreas**

- 0bbc999 track companion scripts and TamIA env

**analysis**

- 394ab7b add rebuttal analyses of PALETTE alone and NGF per method
- 7134fc6 refresh texture fill-swap study on val000 rung-5 retrains
- 7f1ef2f add surroundings texture-complexity vs fill-swap test
- cec7140 add train-me course builder for texture study
- 0585296 turn train-me course into keyboard slideshow
- 47d5043 add difficulty and region-size test of fill-swap sign
- 34d875a add visibility sign rule scored on ladder metrics
- 0fd4330 add visibility example figure for texture study
- 9a8238f add fill-sample and visibility exception figures
- 32c0287 add 7-panel visibility case sheets
- 5abc80b add lblvor mode and no-call zone to visibility rule
- e7274a1 add flat-fill mode to visibility sign rule
- 0690441 add flat-fill visibility case sheets for rule failures
- c6ee698 pick content-rich window in flat-fill case sheets
- 0b1e194 decompose fill swap into noise removal and texture addition
- 2b33ccb add visual proof scale-space strip and Blur IG analyses
- d821e83 add visual proof v7 per-voxel input gradient analysis
- dc6ae7d add seed-centred averaged evidence maps to visual proof v7
- 991ec76 add body-masked maps and quality dependence to visual proof v7
- 6eb0963 add per-task seed-centred evidence maps for visual proof v8
- b7547f2 add no-evidence display mask to v8 task maps
- 6dbe6fa add per-cell panels and forest plot to visual proof v8
- e6a5670 add core-excluded variant to visual proof v8 panels
- d76de01 add matched random control voxels to visual proof v8
- 2ecddd1 add gradient diagnostics and output-side probe to v8
- 1e7298d add all-class logits and heterogeneity drivers to v8 probe
- 961a5cd add between-case drivers and qualitative examples to v8 probe
- e85f4c4 add distance-to-core feature and context test to v8 probe
- b994896 add stress test of distance-to-core context result
- 3183abf add training-image edema visibility check and counterfactuals
- d8b3958 add encoder/decoder linear-probe analysis
- 3d04451 add weight-space and layer-wise CKA analysis
- 6aece61 add per-stage texture content and head alignment analysis
- 07a8f6f add signed input-gradient power spectra analysis
- b2bcebf add case count and tag to texture-content analysis
- aaae6ae add gradient and Seg-Grad-CAM maps for three fill arms
- a7658fa fix non-negative error bars in gradmaps3 report
- c1086c9 rerun HiResCAM and energy analyses with noise-fill control
- 6c6a71c add Fourier and wavelet attribution of input gradient
- 9956769 add fine-scale sensitivity figure, outcome-stratified evidence
- 437ff0f add outcome-stratified per-voxel evidence for v9
- ca46a3d add path-decomposed input gradients for v10
- be7345a add cancellation index to all-paths depth report
- b10011b add all-paths depth profile of the decision per skip
- 6a83a84 add spread/grain per path depth and v11 directional derivative
- 4fc39cd add directional derivative along image fine detail for glioma
- f26f8a7 withdraw degenerate global texture projection after audit
- 7b398a2 add zoomed image-based example figure for v10 path gradients
- 179ffa6 harden grain analysis with multi-sigma, spectrum, random init
- 33e2720 fix random-init arm and add VP_TIME/VP_WAIT overrides
- 7ebb7f7 add per-fold grain and radial spectrum to v10 pass 2
- 31f39dc add v12 texture-switch modulation, patching and erasure
- 82ac377 add v12 texture-switch results for BraTS
- f64c741 add v13 flat-real model stitching on BraTS
- 18c193a add v14 scale-band phase scrambling reliance spectrum
- c95f8cc add v14 energy-map flattening follow-up
- a66170d add v15 texture-evidence maps by activation patching
- 2b936b5 add v16 integrated gradients from texture-flattened twin
- adb6927 add v16 edge-mask control and fold agreement addendum
- a469da3 add v17 LEACE erasure of texture-energy subspace
- e339224 add v18 texture-selective channel ablation
- c410eb8 add v19 DCT kernel spectra and kernel smoothing
- 6f1d2df rescore visibility sign rule with full rung 4.5 coverage
- 5b5030d add pre-registered polarity-flip texture rule script
- 50a99de fix texture_rule_polarity mechanics and record result
- bf315b0 add texture rule search design and hand-feature extraction
- 3908617 add pre-registered interface-sharpness texture rule
- c73cd26 add two-sided H2 to interface rule, make features resumable
- c501353 tighten H2 appearance bound in interface rule
- 49fc8e4 add H2b bound to interface rule
- 0a95650 switch interface rule to pure edge metric
- 90aa3b8 exclude pancreas from texture-rule analyses
- af450ef fix missing-keys report in texture rule search
- 9053940 exclude pelvis and HCC from polarity rule
- fcec668 add pre-registered train/eval edge-gap rule
- d0bcfb2 add edge-gap results and within-task check
- 9407bcb add pre-registered shared-texture rule
- 878f755 add signed shared-texture candidate
- 38fda6b add pre-registered baseline difficulty rule
- cdbcf55 fix shared-texture report string and record results

**atlas-liver-hcc**

- fa98460 add texlite wrappers, TamIA eval job and configs
- 5a95e99 track pipeline scripts after the unarchive

**auglab**

- bf155dc add texlite reduced-Auglab wrappers, TamIA launcher and tooling
- 3acb463 add texlite_keepres reduced-Auglab arm and wrappers
- daed14f restrict reduced-Auglab study to dev settings, add edgeonly

**brats2024-glioma**

- e0d501e add radiomics scripts for rung4-to-5 edema analysis
- 70d6df6 add rung-6 PV t1n wrapper, PV preview and self-tests
- 940e4c6 add exploratory fill-swap rule tuning and figure
- 1a3cad6 point per-label transfer at lblvor rung-4 runs
- 162e05b fix the 12 ruff F821 errors that kept CI red
- 6ec938a track visual-proof and texproof launchers and pilots

**chaos**

- 8f7bcff wire CHAOS FOV-cropped results and cross-dataset ladders
- a926f48 make crop-before-predict standard, add RUNS_FILE mode and 06_36
- 8820405 retrain t2spir rung 5 with AugLab synth-only, cross-dataset

**ci**

- f35ed62 load legacy training scripts from scripts/legacy_synthesis in tests

**cluster**

- 633a57f drop isles2022 from rung5_val000 tooling, merge run lists
- b834208 skip complete predictions in on-harmony best controller
- bec3a9c add pancreas recipe to SRCSM source-matching runner
- 7169895 pack on-harmony rung 4.5 on one TamIA node with pinned AugLab
- 5091cfd add rung 4.5 post-training controllers and flat-fill wrappers
- 3ae839f assert equal units in flat-fill decomposition, verify TamIA pin
- 90a5c31 fix ssh eating run-id list in on-harmony tail script
- ecb52e6 redirect run_job stdin in on-harmony tail script
- 9c7f7f7 document unpinned AugLab finding in TamIA rung 4.5 launcher
- 1b3573f let rung 4.5 post_run skip predict for external predictions
- 179eb98 port SRCSM source-matching liver recipe to TamIA
- 54f12ca add per-epoch latency probe for PALETTE Augmentation
- 540ebd4 add all-running statistic to latency_probe report
- 60f76a5 add per-task matched-run summary to latency_probe report
- 883a71c add early-read tooling for reduced-Auglab arms
- 2517304 add PA variants round 2 wrappers and TamIA pack launcher
- acecdaf add PA variants Vulcan aggregation tail
- f1648d2 tolerate missing training logs in pa_variants launch.sh
- ae03fc0 track fetch_rung7_metrics.sh

**docs**

- b3bb353 document ispy1/acrin6698 onboarding, A-P crop and fold3 fix
- a8ac56b add missing README example image
- 09d2b97 update TamIA migration status and session gotchas
- df67b23 add paper TODO with open questions from review pass
- b77e3a8 note isles2022 training launched, predict and eval queued
- d3cd132 add isles2022 epoch policy and TamIA CPU-job sync notes
- 4d72832 update CLAUDE.md for val000 rungs and on-harmony best checkpoint
- 70f8e9c note on-harmony layout migration done on Vulcan and TamIA
- d2f490a rewrite README with PALETTE-Aug overview, video and results
- 9e597f1 force-add README images despite png ignore rule
- 186b1f6 edit README

**fix video link to README**

- da038f1 

**git**

- e2515c3 add commit/push checks, git hooks and health audit
- 5ea7fe5 add hist.sh, a readable one-line history from refs/notes/history

**healthy-spine-tum**

- 9c0b4cc re-import collaborator results with boundary-distance metric, fix noEM

**isles2022**

- 7664bbb add brain_stroke training task through TamIA pack
- acf1151 set 2000 epochs, orientation QC shows real ADC
- 9237907 drop ADC, add shared verify_heldout and 02_04 wrapper
- 4fd2e07 derive nnUNet_results from PREDICTIONS_ROOT, add sizing probe
- acb9046 add fold-level LPT pack launcher for 60 training folds
- 2cded30 print predicted per-GPU wall in pack launcher
- ebf5c65 queue predict and eval behind training via Slurm deps
- 93c925d clean temp pin on 05_28 failure
- 6776533 add headline configs and record training results
- 3a78589 add ADC polarity control and case-level diagnostics
- 8d89954 archive task as scaffolding reference
- 97380a0 note TamIA scratch copy deleted in archive README

**ispy1**

- d0a0f62 onboard ispy1 and acrin6698 as eval-only breast companions

**ispy2**

- df97354 add A-P crop standard, pool ispy1 and acrin6698 by contrast

**ladder**

- 7c9d12f add rung-6 PV wrappers for all 14 training sets, TamIA launcher
- 7d27a42 add totalseg-pelvic PV rung pack, launcher takes pack names
- 873829e add TamIA rung-5 retrain launcher and predict/eval jobs
- 513ceaf queue rung-6 PV post-training predict and eval
- 6dcb6b3 add PV branch engine hook, pelvis scripts and rung-6 rosters
- b8df2d5 add PV fetch script, open-ms runner and PV registry entries
- ce16c0e make rung-6 PV branch opt-in and register 14 ladders
- 7549399 add final OURS+PV rung wrappers and TamIA pack launcher
- f9b3df6 queue final-rung predict and eval behind training chains
- 99b03b9 select rung-5 checkpoint_best on real validation (val000)
- 3cb6434 retrain rung 5 on val000 for five tasks, add controllers
- face0c0 point pelvis configs at val000 rung 5, guard val100 wrappers
- 7d94970 commit rung-5 retrain and rung-6 PV session work
- 10d4fde point PV branch registry rung 6 at val000 PV runs
- bd1d86c use val000 rung 5 for open-ms T1w and add ladder run keys
- ee0099c point PV registry open-ms T1w rung 6 at val000 PV run
- 365bbf7 use val000 rung 5 for brats2024-glioma ladders
- 5f6d863 point PV registry brats rung 6 at val000 PV runs
- d4f9b64 use val000 rung 5 for on-harmony DWI and add run keys
- 7b56e23 use val000 rung 5 for on-harmony T1w and T2w ladders
- 2bc8641 point PV registry on-harmony rung 6 at val000 PV runs
- a4f246d retrain rung 4 with label_voronoi and add lblvor wrappers
- 210211c repoint first-half ladders to lblvor rung 4 and regenerate
- dc3689a repoint open-ms ladders to lblvor rung 4 and regenerate
- 1b18626 add rung 4.5 flat-fill wrappers and fill verification
- d84c572 repoint brats2024-glioma ladders to lblvor rung 4
- 66ea643 repoint on-harmony ladders to lblvor rung 4
- 02dfb37 add items filter for extra OOD source, liver SRCSM recipe
- 23a9f80 insert flat-fill rung 4.5 before real fill, split Holm families
- f489bb6 add read-only ladder audit and pancreas companions run key

**liverhccseg**

- 7c2eb81 track bidsify, test-set and predict/eval scripts

**lld-mmri-hcc**

- 66b8402 track bidsify, test-set and predict/eval scripts

**lld-mmri-malignant**

- bce3c5e track download, bidsify and predict/eval scripts

**merge**

- 8be9b29 origin/main into paul/v26_6_2 (README video-link fix from GitHub)

**msd-pancreas**

- 407ea24 add CT eval companion and wire into pansegdata tables

**on-harmony**

- 9a3731d resample RAS to native inside predict jobs, refuse off-grid
- 158fa98 make checkpoint_best without TTA the predict/eval default
- 417e5d2 add CLUSTER_ENV override to layout migration script
- 082b003 make layout migration refuse to run twice
- 6c1768a move texlite T1w training to Vulcan with CPU controller

**pansegdata**

- 49bf0e9 add pancreas BIDS build, patient split, nnU-Net conversion
- 409a9a3 scaffold pipeline scripts, add Killarney probe and launcher
- 62d00ee run post-training controller per contrast with 6 h limit
- 9e5fad7 move ladder rungs to Vulcan L40S with pinned AugLab
- 3b93d11 move headline folds to Vulcan, queue controller there
- f9c1fe8 generate results and significance configs from roster
- 1244206 launch rung 4.5 flat-fill folds, add pancreas decomposition

**paper**

- 82e902a cover all 14 ladders in fig:ladder, patient-unit panel test
- 724b6e0 make NGF dose-response figure reproducible, 14-ladder rung table
- 837c329 rewrite results on current data for 8 tasks and 14 ladders
- 266fef8 correct abstract placeholder statements
- 571970c update abstract, spine boundary-distance metric and review pass
- 49bc02d regenerate BraTS per-label supplement, align dataset descriptions
- 42ab6ee fit main text in 8 pages
- f6da014 fix ON-Harmony labels, drop BraTS subset note, add CHAOS FOV TODO
- 37f7541 add CHAOS cross-dataset ladders, state texture effect relatively
- 09278e4 review pass, keep main text at 8 pages
- 2d51476 add FOV-matching paragraph, describe SynthSeg and SRCSM baselines
- 9788efa explain competitors synthesize every iteration in suppl
- c349f26 straightforwardness pass over sentences and first-read flow
- cd63672 add at-a-glance tables for tasks and methods
- d2960ae add pelvis ablation ladder and retrained rungs, light Table 1
- 596ebf0 use task names everywhere, one-column ablation figure
- 23dec62 describe competitors as run by their authors, add limitation
- e523d7d add NGF violin figure and per-task supplementary table
- 68cbb05 compose Fig 1 in radiological view, rework tables and Fig 2
- 3b4d5f3 harmonize supplement task naming, order and statistics section

**paper-scripts**

- cfd14a7 fix post-restructure paths, reuse table design
- 70be5cb use BraTS-GLI spelling in generated tables and figures
- 5d73c7f reorder Fig 1 in ablation order, mark best/significant
- eea786e refine Fig 1 and Fig 2 captions and legend
- d881c4c rewrite introduction and abstract
- 80c9f0f update breast results and SynthSeg baseline descriptions
- cbed453 update results with finished retrains, on-harmony re-eval
- e7cb519 apply 2026-10-07 review to results, abstract and figures
- 9a87d24 align boundary-distance metric captions with regenerated numbers
- 83be833 restore structured abstract, split fill-swap figure
- 62657eb describe FOV crops by purpose only
- d481d01 use one Holm correction family for the fill swap
- af67e01 report SRCSM with and without source matching
- 0d352f1 fill Brain fill-swap numbers from val000 ladders
- 08d010d report SRCSM only with source matching
- ff3505f rewrite method section and Fig 1 step order
- 50a9375 reorder method and Fig 1 to ablation-ladder order
- 6dd4201 update fill-swap figure to lblvor noise fill
- 476f84e share random draws between fill-swap example arms
- b5e8399 revise related work on shape-texture and style randomizat
- fb0b3e1 restructure section 4.1 setup and rename PALETTE alone
- d2e438d split compared-methods opening into per-method sentences
- d20baf8 revise section 4.3 headings and fill-swap wording
- 3e6c38a move fill-swap example figure to a BraTS case
- 9f89c60 show noise, flat and real fill in fill-swap example
- 17421f1 add arrows to real-fill zoom in fill-swap example
- 565ab6d re-measure NGF with training configs for real use
- dd803d0 add flat-fill decomposition of the fill swap
- 94d49b8 record rung 4.5 flat-fill status in TODO
- bed740f update motion video bars to current ladders and results
- c9eb903 track the Remotion motion video project

**repo**

- c3ebb38 archive retired val100 rung-5/6 runs and stale on-harmony metrics
- a449b57 archive legacy eval configs and comparison tables
- 4a23634 untrack internal notes and manuscript; add LICENSE and CI
- cab61e8 untrack remaining internal docs and retire orphaned scripts
- e732dbf untrack paper and dashboard and tidy repo root
- 1b9d5e5 make LICENSE verbatim MIT so GitHub detects it

**results**

- e4a1887 regenerate ladder and fill-swap figures after fold3 fixes
- 88db8f6 use SRCSM source-matched runs in headline configs
- 4369e14 re-include atlas-liver-hcc in meta task heatmap config
- f6fda84 add texlite aggregation configs for headline settings
- 8639496 add texlite_keepres run to texlite aggregation configs
- 1b45cdb regenerate fill-swap figures with flat rung and Pancreas
- 6585be6 add texlite, keepres and edgeonly runs to dev-setting configs
- 59f1e93 regenerate ablation_ladders per-contrast PNGs

**shared**

- fbd5ff3 fix ladder fold3 contamination, add grouped OOD pooling
- a65d41c run meta heatmap cross-task test on patient units
- acf63ca add per-task significance markers and win counts to meta heatmap
- b55c632 add roster-driven predict/eval drivers, isles2022 wrappers
- 77c2751 add create_pipeline_scripts scaffolder and roster selftest
- bd775e6 add eval-only companion toolchain and skills-by-reflex docs
- 70565cd neutralize scaffolder epoch-policy TODO, record Killarney facts
- bd16c2b add SRCSM test-time source matching for predict and eval
- cda2c24 add opt-in ood_only mode to meta task heatmap
- 9eda5bf guard per-dataset DA_WORKERS and nnUNet_compile in train_common
- 3220f9c log main + AugLab commit and dirty count with every training

**totalseg-pelvic**

- 9c66039 add per-label view of the ladders

**totalsegmri-pancreas**

- ce444bd add eval companion and wire into pansegdata tables
- b6e5f43 track 02_03 QC gallery script

</details>

## 2026-09

- Restructure of `datasets/` into `benchmark/02_tasks/<task>/<dataset>` with follow-up path fixes (`47efff1`, `784e6e8`, `07bf146`).
- Canonical `06_evaluate` entry-point numbering applied across datasets; validator and dataset scaffolder updated (`bd4f12f`, `3341e4d`, `a9b20fe`).
- New tasks onboarded: toothfairy2 with hanseg and pddca eval sets, I-SPY2 with duke-breast-mri, totalseg-pelvic; an autopet scaffold was added then removed (`ceb3435`, `79415a5`, `fb73dbb`, `3ff908e`).
- Atlas-liver-hcc archived early in the month (`dd3dbb7`); a PET-task handoff doc was written (`198f8f8`).
- BraTS gained T2f and T1c training modalities, on-harmony gained dwi_ap, and significance testing switched to patient-level units (`066803f`, `fb23960`, `db39bab`, `29dd30e`).
- Breast unilateral-crop evaluation became the standard for duke-breast-mri; ladders pool same-patient pairs (`a6f3d58`, `0ebb3b5`).
- Cluster tooling: TamIA/Killarney relay helpers, pack GPU-map generator, tmux session notes (`c678e2a`, `4f1bdef`, `c1237a2`).

<details><summary>All 124 commits</summary>

**amos**

- c9c054b add config-driven per-organ aggregate for amos and sliver07
- 7c63086 rename output_prefix to 01_results and 02_cross_dataset

**analysis**

- f8c005d add domain-distance and cross-dataset ladder comparison tools

**atlas-liver-hcc**

- 36ba592 extend lld-mmri-hcc phases, fix CE-T1w pooling

**autopet**

- 4b6b9a4 onboard AutoPET PET+CT as new training task
- 3ff908e remove scaffold after task abandoned

**brats2024-glioma**

- 066803f add T2f/FLAIR training modality with 3-rung ladder
- 3b5d5b5 fix nnUNet_results override clobbered on TamIA
- 079f39d add TamIA pack launcher for T2f suite and ladder
- 7b7fc09 add T2f predict and eval chain for TamIA
- 4d53b46 fix T2f predict-job pre-flight glob
- 09289bf run T2f eval folds sequentially with 96G memory
- cb49cc9 add T2f aggregation, significance and 3-modality table
- fb23960 add T1c as fourth training modality
- 08b3b70 add T1c dry-run guard test, CPU-only preprocess
- 9e12862 add level-1 texture-analysis pipeline for fill-swap
- fd84c7f fix numbering collision, mis-homed script, dead cruft
- 14e31d7 make 4-modality combined table canonical
- d46d901 remove superseded 3mod and 4mod configs
- 0f38ca0 harmonize 06_evaluate entry-point numbering
- 155ebb2 add missing 06_03 significance wrapper
- 570308c rename output_prefix to 01_results/02_cross_dataset
- f61c5ff fix stale tamia_env.sh hop after restructure

**chaos**

- 80c2aff apply pooled-column display and rename amos mri column to t2w
- 7b2af45 fix 06_evaluate numbering collisions, dedupe backup dirs
- ee9b35d finish 06_evaluate harmonization for amos and sliver07
- 53113ee fix dirname/$0 resolved-after-cd path bug and template

**ci**

- 59fd81c expand auto-mode Slurm and git tooling allowlist

**cluster**

- c678e2a add TamIA/Killarney relay helpers and per-dataset env overrides

**docs**

- 198f8f8 add handoff for new PET task
- 2aea250 require two test modalities in PET handoff, pick HECKTOR
- 86caa17 correct PET handoff to two test modalities, not training
- f35b7a7 allow combining single-modality datasets in PET handoff
- 79e42d1 scope PET handoff to HECKTOR with CT-arm go/no-go gate
- b9c3bae record TamIA, pack and onboarding gotchas in CLAUDE.md
- c1237a2 pin session cleanup and document per-session tmux socket fix
- e60a6ad record breast unilateral-crop standard and cross-dataset policy
- f73465e fix stale on-harmony results-layout note in CLAUDE.md
- a9b20fe document canonical 06_evaluate numbering and shared-script rule
- 7c239e8 add README

**duke-breast-mri**

- e16bcfe onboard as cross-dataset eval companion to ispy2
- 7c361cc derive unilateral crop and score bilateral-FOV skew
- a6f3d58 switch standard evaluation to unilateral crop
- 20761c4 fix env.sh doc contradiction, add aggregate wrapper
- 9bbef5c add missing 06_04 combined summary wrapper

**hanseg**

- fec75ad add FOV-matched CBCT-to-CT test set for toothfairy2
- a72dfa8 stop env override clobbering caller variables
- d6940aa add MR-T1 arm with GT propagated from CT
- 143ea0a make propagated MR ground truth reproducible and run refinement
- b92e1bc disable propagated-GT MR arm
- a98611d add MRI out-of-distribution arm in CT frame
- d11a01f correct mandible label claim, re-score mandible-only
- 002e7da add TamIA node-pack predict launcher for CT and MR arms
- f6be6b1 document recovery script and config-naming divergence
- a493023 fix stale 02_metrics_mandible_only path in OOD config

**healthy-spine-tum**

- 0b96e55 add own-model significance configs
- 4c48877 add standard eval-layer entry points

**infra**

- e486905 auto-discover nnU-Net trainer shims in venv setup
- 4f1bdef add TamIA pack GPU-map generator, pilot on brats t2f

**ispy2**

- 79415a5 onboard I-SPY2 as breast task training set
- e1f1670 add own-model significance driver and configs
- 273c46d add level-1 breast texture-analysis scripts
- a226aae harmonize 06_evaluate entry-point naming and numbering

**ladder**

- 11ab3b0 make plots OOD-only to match paper figure
- e247924 add two new tasks to fig:ladder with 2x3 JSON-driven layout
- e059aa8 add external cohort to breast OOD pool
- 0ebb3b5 pool same-patient pairs across contrasts before testing

**on-harmony**

- db39bab onboard dwi_ap as third training modality
- 2306fd8 merge convert_to_nnunet into 02_nnunet, fix stale tests
- 03788ac extract inline predict into 05_predict stage
- be58526 renumber 06_evaluate aggregate and combined-summary scripts
- 46a9414 add missing 06_03 significance wrapper
- 9d14df9 rename headline output_prefix to 01_results

**open-ms**

- 648b3e9 harmonize 06_evaluate entry-point numbering
- 639d107 rename cross-dataset output_prefix to 02_cross_dataset

**paper**

- de17a3d add dental and breast tasks to results and ladder
- 04108d1 update roster-dependent sections after roster change
- 8c999d4 rebuild PDF
- 892daf6 verify two attributions and Decathlon support in bib and suppl
- fb6bbdd rebuild PDF
- 407d48b add 3-contrast BraTS overlay illustration
- 37a442f export FRQ illustration panels as separate images
- 3686c40 export FRQ illustration panels at 8x
- 0e676b4 use plain T2-FLAIR panels at 1:1 without overlay
- a54e72f fix t2flair_gt panel ground truth

**paper-scripts**

- 28dcaae add reverse significance test for methods leading ours
- cf10763 read dissociation table from engine numbers, add 2 tasks
- 0a7b700 read raw fold-level data for contrast_groups significance

**pddca**

- 79a4c00 onboard as toothfairy2 second independent CT eval cohort

**repo**

- dd3dbb7 archive atlas-liver-hcc and revert paper additions
- e996d7b gitignore xlsx files
- f6b8bb3 stop tracking non-README markdown notes and reports
- 47efff1 restructure datasets/ into benchmark/02_tasks/<task>/<dataset>
- 784e6e8 fix remaining broken path references after restructure
- 07bf146 fix bare .parent chain path missed after restructure
- bd4f12f enforce canonical 06_evaluate numbering in validator
- 3341e4d scaffold canonical 06_evaluate skeleton in dataset creator

**results**

- 01c9312 add 7 tasks to meta heatmap and cross-task ladders
- 687cc08 add totalseg-pelvic as 8th meta-heatmap task

**shared**

- 99ddcf9 contrast_groups controls displayed columns, not only pooling
- adf9823 justify sign-flip macro-delta test and correct its description
- f025ee3 fix wrong nnUNet checkpoint dataset id in predict_common
- 29dd30e sign-flip significance uses patient as independence unit
- f1d3a9b honor sig_ref config key in significance_from_config
- bb37b71 match per-case label pooling to displayed fold-then-label table
- fcdf231 resolve relative CFG override against script directory
- bc47079 fix aggregate_from_config docstring to 01_results prefix
- 8bb82cc fix output_dir ignored in single-source aggregate mode

**toothfairy2**

- ceb3435 onboard maxillofacial CBCT task
- 5bf2252 add post-training predict, evaluate and aggregate chain
- 5bae446 anchor orchestrator config patch to method and timestamp
- c696da7 add whole-node predict and evaluate job
- 1c60e53 add cross-dataset and combined tables
- ed7b5d3 use PROJECT_ROOT instead of METRICS_ROOT in configs
- 1630d34 add results notes for ladder rung 4 to 5
- 0b716b7 add paper-facing provenance and results record
- 0fcd2b6 add mandible-union table scoring every column on one label
- 7483192 score combined table on mandible union, note AugLab CT drop
- fac1f0d make mandible-only the sole standard, retire union

**totalseg-pelvic**

- fb73dbb onboard CT+MRI pelvic musculoskeletal task
- e9351de fix non-orthonormal affine crash, add TamIA env
- cb6e513 wire real RUN_IDs and ladder methods into eval configs
- 3640d63 keep ablation-ladder rungs out of headline tables
- 40969ef guard sig_ref against removal in configs

</details>

## 2026-08

- More eval-only companions for chaos models (kidney-t2w, cirrmri-liver) with TamIA pack jobs, plus a FOV hard-crop driver (`d93dd0a`, `9aff1bb`, `6431642`).
- Picai-prostate added as a training task, then untracked when abandoned (`9fd4ee7`, `fff19e8`).
- Shared combined-modality summary and task-level meta heatmap added, plus a cross-cluster monitoring dashboard (`32500ee`, `f88969e`).
- Atlas-liver-hcc onboarded with lld-mmri-hcc and liverhccseg companions, then archived together with excluded evaluation sets (`797a234`, `e9a7188`, `b1202b4`, `dd3dbb7`).
- Shared ladder engine and `contrast_groups` pooling introduced (`982d5f5`, `f96f243`).
- Healthy-spine-tum onboarded; paper scripts reorganised and the manuscript reframed (`5944331`, `19da397`, `ae700a9`).

<details><summary>All 48 commits</summary>

**atlas-liver-hcc**

- 797a234 onboard HCC liver tumour segmentation dataset
- 0659929 add BIDS tree and TamIA pack launch infra
- 55d2530 add TamIA pack predict and evaluate script
- ceb02c0 add val100 mirror and wire into meta heatmap

**brats2024-glioma**

- fba482b add per-label T1n/T2w transfer breakdown
- 9d189f6 document BraTS-SSA2024 removal from combined config

**chaos**

- 6431642 exclude kidney-t2w from roll-ups, add FOV hard-crop driver
- 0be30f7 exclude cirrmri-liver from roll-ups over fragmentation

**cluster**

- 665236c add TamIA cross-dataset env for kidney-t2w

**docs**

- b419ad8 retire two-pillar framing and note Killarney path drift

**healthy-spine-tum**

- 5944331 onboard spine segmentation dataset

**infra**

- f88969e add cross-cluster monitoring dashboard

**kidney-t2w**

- d93dd0a onboard as eval-only dataset for chaos models
- e4e396e add TamIA pack predict and evaluate job
- 3871d8a run TamIA pack, wire tables, fix summarize_fold merge
- e02889c add geometry ablation and acquisition-plane investigation

**ladder**

- 5fe5991 add on-harmony T1w arm and brats/chaos ladder summaries
- 982d5f5 add shared ladder engine and second-modality ladders
- 8e223df add atlas-liver-hcc cross-dataset ladder

**liverhccseg**

- a070356 onboard as eval-only dataset for atlas-liver-hcc
- 2a3bcec fix orientation by reorienting to LPS

**lld-mmri-hcc**

- e9a7188 onboard as cross-contrast eval set for atlas-liver-hcc

**on-harmony**

- 1a12360 remove dead first-generation train and eval scripts

**paper-scripts**

- 19da397 reframe manuscript around texture-first narrative
- 0e7c467 generate method figure panels instead of static PNG
- 5dae3fc trim related work, method and supplement framing
- 11a6da5 add Liver-HCC results and recategorize dissociation
- c6c4109 add atlas-liver-hcc to ablation dissociation
- d789945 redesign ladder figure with panel-pooled significance

**picai-prostate**

- 9fd4ee7 add csPCa lesion training dataset and TamIA packs
- 02c328b add eval configs and cap pack density at 2 folds
- 0332c10 record H100 timings and trim pack chains
- a20ab21 add predict and evaluate drivers for trained folds
- fff19e8 untrack abandoned dataset

**repo**

- b36823c remove stale .gitkeep placeholders in amos and sliver07
- b1202b4 archive excluded evaluation datasets
- bb9c93a route dataset pipelines through renamed BIDS leaf dirs
- f64d0b1 remove stale LaTeX template backup directory
- 001775e clean up LaTeX project and untrack build logs
- ae700a9 reorganize paper into scripts, notes and LaTeX project
- c90f6b0 gitignore log files and .DS_Store

**results**

- e760c1c add FOV hard-crop vs masked-eval comparison

**shared**

- 32500ee add combined-modality summary and task-level heatmap
- f96f243 add contrast_groups pooling and single-modality tasks
- f7e76be remove MSD-Spleen and MS cohorts from cross-dataset pooling
- edddb7b add atlas-liver-hcc to the cross-task meta heatmap
- 8a9f921 fix dirname-after-cd bug in evaluate_run scripts
- 97f8df5 tighten dataset structure validator

</details>

## 2026-07

- On-harmony texture generation/metric scripts and 31-class histogram coverage analysis, followed by open-ms coverage (`9ee5654`, `e4b9b4f`, `2176a70`).
- Shared layer consolidated: stat tests, run-all training launcher, `aggregate_from_config` moved into the shared evaluate layer, train/predict/aggregate drivers refactored (`b901384`, `e4378af`, `fbb3531`).
- Open-ms FLAIR suite and causal-ablation arms (noisefill, k-means, label remap, affine remap), with a Killarney handoff doc (`e8684c3`, `604e46c`, `c0cf162`, `991afc3`).
- Chaos T1in ablation ladder and the DualVal trainer ported to chaos and open-ms (`c091153`, `951a625`).
- First eval-only companions onboarded: mslesseg, brats-ssa2024, ms3seg, msd-spleen, cirrmri-liver (`8d86afd`, `49a0c53`, `5fc8c43`, `c077dfb`, `232e18a`).
- Manuscript revision commits and repo hygiene (dataset skeleton placeholders, unused datasets removed) (`ef4d5f7`, `e028ebd`, `4942e5d`).

<details><summary>All 86 commits</summary>

**amos**

- 0b3a6fb add chaos SRCSM and val000 predict scripts

**analysis**

- 9ee5654 add on-harmony texture generation scripts
- 05b5a46 update on-harmony texture literature review
- 7276492 add on-harmony texture metric scripts
- ad0e911 add on-harmony data generation report
- 2b539df fix on-harmony texture metrics and generation
- bf48848 update on-harmony texture metrics
- 1a24499 simplify on-harmony texture metrics
- ac74cf3 add no-blur on-harmony volume generation
- ae9e232 update texture literature review and plots
- b56c83d update on-harmony generation report
- e4b9b4f add on-harmony 31-class histogram coverage
- 4f30560 update histogram coverage literature review
- b237556 precompute distances for fast coverage bootstrap
- 2176a70 add open-ms histogram coverage analysis
- f09763c use lesion-only labels in open-ms coverage
- ca85da9 add new texture metric and update configs

**brats-ssa2024**

- 49a0c53 onboard as eval-only dataset for brats2024-glioma

**brats2024-glioma**

- a7361ca add SRCSM train and predict scripts
- 5d45398 add val000 train and predict scripts
- 97a2b58 add combined cross-dataset configs with brats-ssa2024

**chaos**

- cc9b66e fix t1in srcsm train script
- 2d773d2 add evaluation configs and update scripts
- c091153 add T1in ablation ladder with dual-validation trainer
- 59c37d1 materialize DualVal val000 and val100 as separate runs
- c055c6c remove dualval naming from T1in ladder scripts
- 4b8cef8 update DualVal trainer for val000-primary naming
- b8c1661 add T1in ablation branch configs and significance wrapper

**cirrmri-liver**

- 232e18a onboard as eval-only dataset for chaos models
- 9aff1bb pack two predict tasks per GPU and wire into tables

**cluster**

- 3b99ada add TamIA cross-eval env for brats-ssa2024
- 3b3686d add TamIA cross-eval env for ms3seg

**docs**

- a26a43e update CLAUDE.md and Killarney handoff

**infra**

- 4e6ca87 stop env.sh from clobbering variables in cross-eval predict

**ladder**

- c0cf162 add kmeans rung without label remap
- 991afc3 add affine-remap ablation arm for chaos and open-ms

**merge**

- 27e020d paul/v26_6_2 remote into local branch
- ff842af remote paul/v26_6_2 into local branch
- aa5171e remote paul/v26_6_2 into local branch
- 05ae42f remote paul/v26_6_2 into local branch
- 2a08004 remote paul/v26_6_2 into local branch
- 8b67fef Killarney origin config additions into Vulcan paper revision
- 8f4b481 remote paul/v26_6_2 into local branch

**ms3seg**

- 5fc8c43 onboard as eval-only dataset for open-ms models

**msd-spleen**

- c077dfb onboard as eval-only dataset for chaos models
- dc0e500 add TamIA predict and evaluate pack, fix relative path

**mslesseg**

- 8d86afd onboard as eval-only dataset for open-ms models
- 18af1b5 fix RUN_IDs to existing checkpoints and add TamIA env
- f348fad add open-ms T1w-trained cross-eval branch

**on-harmony**

- 961846c add noisefill train025 val100 training scripts

**open-ms**

- e8684c3 add Killarney handoff doc
- 498da64 fix training scripts
- 3d1e147 add noisefill v2 texture experiment scripts
- 604e46c add noisefill alone texture ablation arms and ladder
- d6ae8cf add noisefill-alone predict scripts and ablation config
- f82bbbe fix stale fold-count comment in FLAIR srcsm wrapper
- b851e54 update aggregation configs
- 78e4515 update FLAIR aggregation configs
- fea535c rename FLAIR causal-ablation arms for clarity
- 8180b04 fix stale ablation names in FLAIR ladder script
- ed2fbee fix stale run ids in FLAIR ablation configs
- 26b9703 add train050 val000 arms for FLAIR ablation rungs
- 81bb6a9 split FLAIR ablation into four branch configs
- b6e9788 add kmeans rung to val100 and create val000 configs
- 951a625 port DualVal trainer and add chaos T2spir ladder arms
- 8ef4097 add val100 predict script and deployed-config ablation
- 9e270bb add mslesseg combined cross-dataset aggregation config
- 8ba73b6 add ms3seg as third source in cross-dataset tables

**paper-scripts**

- eccc99d update paper sources
- ef4d5f7 apply 41-comment revision to manuscript
- edd9f5a recompile manuscript after merge
- 5ffd43a track method pipeline figure
- 41fcd31 recompile manuscript after cleanup

**repo**

- 6830d33 update gitignore
- 06f6f54 remove misplaced METRICS_ROOT metrics files
- daeee8e exempt paper figure PNGs from gitignore
- e028ebd seed dataset skeleton with .gitkeep placeholders
- 4942e5d remove unused datasets

**results**

- ea381e0 add significance tables from METRICS_ROOT run

**shared**

- 4756937 fix evaluation and add FOV utilities
- b901384 add stat tests and significance-from-config updates
- e4378af add run-all training launcher and open-ms FLAIR suite
- 11122e0 move aggregate_from_config into shared evaluate layer
- fbb3531 refactor train, predict and aggregate drivers
- 04bca9d fix aggregate_from_config
- 79f983d update predict driver and open-ms checkpoint-final config

</details>

## 2026-06

- Project structure reorganised; nnU-Net trainer shim package added for brats2024-glioma (`8e22c02`, `cd35690`).
- New datasets and scripts added; scripts made flexible for Vulcan (`5d5de4b`, `89be74d`).
- Hardcoded dataset names in amos predict scripts fixed (`9630b94`).

<details><summary>All 6 commits</summary>

**amos**

- 9630b94 fix hardcoded names in predict scripts

**brats2024-glioma**

- cd35690 add trainers package for nnU-Net shim

**infra**

- 89be74d make scripts flexible for Vulcan

**repo**

- 8e22c02 reorganize project structure
- 5d5de4b add new datasets and scripts

**shared**

- 8174e11 add common scripts and new datasets

</details>

## 2026-04

- Synthesis v18 model configs and variants, with logging updates (`072fc20`, `da7f9f6`).
- First nnU-Net implementation added (pre-cleanup) (`f270290`).

<details><summary>All 3 commits</summary>

**infra**

- f270290 add nnU-Net implementation (pre-cleanup)

**synthesis**

- 072fc20 add v18 model configs
- da7f9f6 update logging and add v18 variants

</details>

## 2026-03

- Initial project structure committed (`4471f59`).
- Segmenter training/evaluation updates (v4) with train/test leakage prevention (`6ef475d`, `f551ef6`).
- Lightning and Kornia implementation, followed by repeated training-loop and augmentation speed-ups (`3c25d84`, `5726f0e`, `e6f8326`).
- Synthesis fix for v6, BigAug baseline augmentations, v17 model configs and a refactor of versions and evaluation script (`b8c053f`, `89c54c2`, `7ee36b7`, `fde754e`).
- Architecture, performance and knowledge-transfer docs written (`a65f497`).

<details><summary>All 16 commits</summary>

**docs**

- a65f497 add architecture, performance and knowledge-transfer docs
- 813b6bb update docs and generator/segmenter run scripts

**infra**

- 3c25d84 add Lightning and Kornia implementation
- 5726f0e speed up Kornia augmentation pipeline
- 4ddf4dd speed up training loop
- 878d805 speed up segmenter trainer
- 80b4305 speed up training and clean checkpoint saving
- e6f8326 speed up training and clean up code

**repo**

- 4471f59 add initial project structure

**results**

- 7737a7a add v4 evaluation results

**shared**

- 6ef475d prevent train/test leakage with generated data
- f551ef6 update evaluation and segmenter training (v4)

**synthesis**

- b8c053f fix performance drop in v6
- 89c54c2 add BigAug baseline augmentations
- 7ee36b7 add v17 model configs and clean up
- fde754e refactor versions and evaluation script

</details>
