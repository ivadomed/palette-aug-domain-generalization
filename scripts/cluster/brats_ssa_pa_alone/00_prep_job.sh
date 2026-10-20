#!/bin/bash
# CPU job body (submit with 00_submit_prep.sh): rebuild the BraTS-SSA 2024 test inputs, prove their
# orientation, prove the rebuilt GT is the one that scored the 2026-07-29 metrics, and build the
# SRCSM source-matched inputs for the 4 Glioma training contrasts.
#   1. archived chain, run inline: 00_00 BIDSify (95_Glioma only) -> 03_00 RAS->LPS -> 05_00 imagesTs/labelsTs,
#      then reorient_inputs.py -> LAS (the glioma training arrays' convention; the archived LPS target was wrong)
#      (2026-10-20: the rebuilt LPS labels reproduced the 07-29 metrics exactly, 1140/1140 rows: GT parity OK)
#   2. qc_orientation.py on the final inputs (headers, grid-independent sign test vs the glioma test set,
#      label-in-brain, gallery); exits non-zero on a suspected flip
#   4. srcsm_match_setting.sh glioma_<tc> for tc in t1n t2w t2f t1c on the 4 SSA input dirs. SETTING is
#      exactly the paper's (glioma_<tc>), so the existing $SCRATCH/srcsm_match/glioma_<tc>/average_cdf.json
#      is reused, never recomputed.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
unset nnUNet_raw nnUNet_preprocessed nnUNet_results PREDICTIONS_ROOT METRICS_ROOT
export RUN_JOB_INLINE=1
ROOT="$PWD"
A="${ROOT}/benchmark/03_archive/brats-ssa2024"
S="${A}/5_scripts_brats-ssa2024"
RAW="${A}/2_nnUNet_brats-ssa2024/raw"
OUT="${SCRATCH:?}/review_20261020/brats_ssa"
G="${ROOT}/benchmark/02_tasks/brain_tumor/brats2024-glioma"
PY="${ROOT}/.venv/bin/python"
step() { echo; echo "=== [$(date '+%F %T')] $*"; }

LPS="${A}/2_nnUNet_brats-ssa2024/raw_LPS_AP_flipped_superseded_20261020"
if [ "$(ls "${RAW}/imagesTs_t2f" 2>/dev/null | wc -l)" != 95 ]; then
  step "1. rebuild inputs: archived chain (BIDSify -> LPS -> imagesTs) then LAS, the array convention of the glioma training data"
  # The archived chain writes LPS (it assumed brats2024-glioma was LPS; the glioma nnU-Net arrays are LAS,
  # measured 2026-10-20: LPS inputs are A-P reversed vs the training arrays, see qc/lps vs qc/las).
  if [ "$(ls "${LPS}/imagesTs_t2f" 2>/dev/null | wc -l)" != 95 ]; then
    mkdir -p "${RAW}"; mv "${RAW}" "${LPS}"; mkdir -p "${RAW}"
    ( export nnUNet_raw="${LPS}"; bash "${S}/00_utils/00_00_ingest_and_bidsify.sh"; bash "${S}/03_preprocess/03_00_reorient_to_lps.sh" --trees bids
      bash "${S}/05_predict/05_00_build_test_inputs.sh" ) || exit 1
    mv "${RAW}"/* "${LPS}"/ 2>/dev/null || true
  fi
  "${PY}" "${ROOT}/scripts/cluster/brats_ssa_pa_alone/reorient_inputs.py" "${LPS}" "${RAW}" LAS
fi
for it in t1n t1c t2w t2f; do echo "  ${it}: $(ls "${RAW}/imagesTs_${it}" | wc -l) images, $(ls "${RAW}/labelsTs_${it}" | wc -l) labels"; done

step "2. orientation QC (final inputs)"
"${PY}" "${ROOT}/scripts/cluster/brats_ssa_pa_alone/qc_orientation.py" "${RAW}" "${G}/2_nnUNet_brats2024-glioma/raw" \
  "${G}/4_splits_brats2024-glioma/test_cases.json" "${OUT}/qc/final" || { echo "!! orientation QC FAILED (see ${OUT}/qc/final/orientation_qc.md)"; exit 1; }

step "4. SRCSM source matching (reuse the paper's glioma_<tc> CDFs)"
TR="${G}/2_nnUNet_brats2024-glioma/raw"
for spec in "t1n:Dataset051_BraTS2024GliomaT1n" "t2w:Dataset052_BraTS2024GliomaT2w" "t2f:Dataset053_BraTS2024GliomaT2f" "t1c:Dataset054_BraTS2024GliomaT1c"; do
  tc="${spec%%:*}"; ds="${spec##*:}"
  [ -s "${SCRATCH}/srcsm_match/glioma_${tc}/average_cdf.json" ] || { echo "!! missing paper CDF for glioma_${tc}"; exit 1; }
  md5sum "${SCRATCH}/srcsm_match/glioma_${tc}/average_cdf.json"
  if [ "$(ls "${RAW}/imagesTs_t2f_srcmatch_${tc}/" 2>/dev/null | command grep -c '_0000.nii.gz$')" = 95 ] && \
     [ "$(ls "${RAW}/imagesTs_t1n_srcmatch_${tc}/" 2>/dev/null | command grep -c '_0000.nii.gz$')" = 95 ]; then echo "  ${tc}: already matched"; continue; fi
  EXCLUDE_CASES="${G}/4_splits_brats2024-glioma/test_cases.json" \
    bash "${ROOT}/benchmark/00_commun_scripts/00_02_predict/srcsm_match_setting.sh" "glioma_${tc}" "${tc}" "${TR}/${ds}/imagesTr" mr \
      "${RAW}/imagesTs_t1n:mr" "${RAW}/imagesTs_t1c:mr" "${RAW}/imagesTs_t2w:mr" "${RAW}/imagesTs_t2f:mr" 2>&1 | command grep -E "KS|average CDF|DONE|rror" || true
  md5sum "${SCRATCH}/srcsm_match/glioma_${tc}/average_cdf.json"
done
for tc in t1n t2w t2f t1c; do for it in t1n t1c t2w t2f; do
  echo "  srcmatch_${tc} ${it}: $(ls "${RAW}/imagesTs_${it}_srcmatch_${tc}/" 2>/dev/null | command grep -c '_0000.nii.gz$')"
done; done
step "PREP DONE"
