#!/usr/bin/env bash
# GPU job for the visual-proof PILOT (Seg-Grad-CAM-style attribution of the flat-fill vs real-fill models on the same test
# image; see visual_proof_pilot.py). One L40S, < 1 h. Mirrors run_intervention_predict.sh's env block.
#   bash run_visual_proof_pilot.sh [--cells openms,pelvis] [--case <id>]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
LOGS_DIR="${SCRATCH:-/scratch/${USER}}/visual_proof_pilot"; mkdir -p "${LOGS_DIR}"
SCRIPT="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_pilot.py"
run_job --name visual_proof_pilot --gpus 1 --cpus 4 --mem 32G --time 01:00:00 --wait \
  --log "${LOGS_DIR}/pilot_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    export PYTHONPATH='${PYTHONPATH:-}'
    export CUDA_VISIBLE_DEVICES=0
    cd '${PROJECT_ROOT}'
    .venv/bin/python ${SCRIPT} $*
  "
