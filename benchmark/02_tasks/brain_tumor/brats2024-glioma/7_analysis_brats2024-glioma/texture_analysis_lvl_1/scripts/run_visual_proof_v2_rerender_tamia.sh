#!/usr/bin/env bash
# CPU pass on TamIA: rebuild the v2 sheets (visible predictions, short titles) + per-lesion tables with 4 references + per-case coverage
# from the saved <case>_maps.npz, then the stats/plots pass.   bash run_visual_proof_v2_rerender_tamia.sh [cells]   (default: all)
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
CELLS="${1:-openms_t2w,openms_t1w,brats_t1n,brats_t2f,pelvis_ct}"
VP_OUT="${VP_OUT:-}"   # export VP_OUT=.../visual_proof_v2_brats_full to re-render the full-BraTS run
LOGS_DIR="${SCRATCH}/visual_proof_v2${VP_OUT:+_$(basename ${VP_OUT})}"; mkdir -p "${LOGS_DIR}"
SCRIPT="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_v2.py"
run_job --name vp2_rerender --gpus 0 --cpus 8 --mem 64G --time 01:30:00 \
  --log "${LOGS_DIR}/rerender_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    export VP_OUT='${VP_OUT}'
    export VP_BRATS_RAW='${SCRATCH}/brats2024-glioma/2_nnUNet/raw/Dataset052_BraTS2024GliomaT2w'
    cd '${PROJECT_ROOT}'
    .venv/bin/python ${SCRIPT} --rerender --cells ${CELLS}
    .venv/bin/python ${SCRIPT} --stats-only
  "
