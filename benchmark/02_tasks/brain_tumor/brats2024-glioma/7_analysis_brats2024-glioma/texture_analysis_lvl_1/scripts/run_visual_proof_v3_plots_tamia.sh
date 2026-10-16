#!/usr/bin/env bash
# CPU-only plots/stats pass for visual_proof_v3_activations.py on TamIA (never a GPU node).   bash run_visual_proof_v3_plots_tamia.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
LOGS_DIR="${SCRATCH}/visual_proof_v3"; mkdir -p "${LOGS_DIR}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"
run_job --name vp3_plots --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --log "${LOGS_DIR}/plots_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export VP3_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_v3_activations'
    export VP_BRATS_RAW='${SCRATCH}/brats2024-glioma/2_nnUNet/raw/Dataset052_BraTS2024GliomaT2w'
    cd '${PROJECT_ROOT}'
    .venv/bin/python ${A}/scripts/visual_proof_v3_activations.py --plots
  "
