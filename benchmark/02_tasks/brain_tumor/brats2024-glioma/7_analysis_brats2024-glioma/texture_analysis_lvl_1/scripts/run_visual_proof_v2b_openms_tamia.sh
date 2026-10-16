#!/usr/bin/env bash
# v2b (2026-10-11): Open-MS cells only, on the optimised script (per-lesion zones on cropped boxes); job 520955 did BraTS + pelvis but its Open-MS pass was too slow (full-volume EDT per lesion x arm x stage). Runs the stats pass at the end over ALL cells.
# TamIA job for visual_proof_v2.py: one whole h100 node (h100 is by-node only), the 5 cells spread over the 4 GPUs, then a CPU
# stats/plots pass. TamIA keeps bulk data on $SCRATCH (open-ms / totalseg-pelvic mirror the Vulcan layout; brats2024-glioma uses
# $SCRATCH/brats2024-glioma/{2_nnUNet/raw,8_results/01_predictions}). TamIA's venv trainer shims are stale (pre-restructuring path),
# hence every dataset's 5_scripts_* on PYTHONPATH.   bash run_visual_proof_v2_tamia.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_GPU_TYPE=h100 RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
LOGS_DIR="${SCRATCH}/visual_proof_v2"; mkdir -p "${LOGS_DIR}"
SCRIPT="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_v2.py"
T="benchmark/02_tasks"
run_job --name visual_proof_v2_openms --gpus 4 --time 02:00:00 \
  --log "${LOGS_DIR}/v2_tamia_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    export PYTHONPATH='${PROJECT_ROOT}/${T}/brain_ms/open-ms/5_scripts_open-ms:${PROJECT_ROOT}/${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma:${PYTHONPATH:-}'
    export VP_OPENMS_ROOT='${SCRATCH}/open-ms'
    export VP_PELVIS_ROOT='${SCRATCH}/totalseg-pelvic'
    export VP_BRATS_RAW='${SCRATCH}/brats2024-glioma/2_nnUNet/raw/Dataset052_BraTS2024GliomaT2w'
    export VP_BRATS_PRED='${SCRATCH}/brats2024-glioma/8_results/01_predictions/brats2024_glioma_model/t2w/auglab'
    cd '${PROJECT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ${SCRIPT} --cells openms_t2w > ${LOGS_DIR}/cell_openms_t2w.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ${SCRIPT} --cells openms_t1w > ${LOGS_DIR}/cell_openms_t1w.log 2>&1 &
    wait
    tail -n 30 ${LOGS_DIR}/cell_*.log
    .venv/bin/python ${SCRIPT} --stats-only
  "
