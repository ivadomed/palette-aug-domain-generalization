#!/usr/bin/env bash
# Vulcan CPU pass: rebuild the v2 sheets + per-lesion/per-case tables from the <case>_maps.npz copied back from TamIA, then the stats pass.
#   bash run_visual_proof_v2_rerender.sh [cells]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
CELLS="${1:-openms_t2w,openms_t1w,brats_t1n,brats_t2f,pelvis_ct}"
LOGS_DIR="${SCRATCH}/visual_proof_v2"; mkdir -p "${LOGS_DIR}"
SCRIPT="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_v2.py"
run_job --name vp2_rerender --gpus 0 --cpus 8 --mem 64G --time 02:00:00 \
  --log "${LOGS_DIR}/rerender_vulcan_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    cd '${PROJECT_ROOT}'
    for c in \$(echo ${CELLS} | tr , ' '); do .venv/bin/python ${SCRIPT} --rerender --cells \$c > ${LOGS_DIR}/rerender_vulcan_\$c.log 2>&1 & done; wait
    tail -n 5 ${LOGS_DIR}/rerender_vulcan_*.log
    .venv/bin/python ${SCRIPT} --stats-only
  "
