#!/usr/bin/env bash
# TamIA submit wrapper for the visual-proof PILOT (moved from Vulcan 2026-10-10 after 3 h pending on the L40S queue).
# Data on TamIA lives under $SCRATCH/<dataset>/... (not in the repo) -> VP_*_ROOT overrides; whole-node h100 class,
# per CLAUDE.md "TamIA": RUN_JOB_GPU_TYPE=h100 RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G (GiB), no --partition.
# h100 is allocated BY NODE ONLY (sbatch rejects --gres=gpu:h100:1), so the job takes the 4-GPU node and runs the two cells in
# parallel on GPU 0 / GPU 1 (each cell is a ~10 min single-GPU task); the 2x3 summary figure is assembled by a third, CPU-only pass.
#   bash run_visual_proof_pilot_tamia.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_GPU_TYPE=h100 RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
LOGS_DIR="${SCRATCH}/visual_proof_pilot"; mkdir -p "${LOGS_DIR}"
SCRIPT="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/visual_proof_pilot.py"
run_job --name visual_proof_pilot --gpus 4 --time 01:00:00 \
  --log "${LOGS_DIR}/pilot_tamia_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    # TamIA's venv shims (OpenMSTrainers.py, TotalsegPelvicTrainers.py) are STALE copies with the pre-2026-09-27 'datasets/<ds>/5_scripts_*'
    # path, so NNUNET_PROJECT_ROOT alone does not resolve the trainer packages there (job 520732 failed on exactly this); put both
    # datasets' 5_scripts dirs on PYTHONPATH like each dataset's own env.sh does. No venv edit: other sessions' jobs import from it.
    export PYTHONPATH='${PROJECT_ROOT}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms:${PROJECT_ROOT}/benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PYTHONPATH:-}'
    export VP_OPENMS_ROOT='${SCRATCH}/open-ms'
    export VP_PELVIS_ROOT='${SCRATCH}/totalseg-pelvic'
    cd '${PROJECT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ${SCRIPT} --cells openms > ${LOGS_DIR}/cell_openms.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ${SCRIPT} --cells pelvis > ${LOGS_DIR}/cell_pelvis.log 2>&1 &
    wait
    cat ${LOGS_DIR}/cell_openms.log ${LOGS_DIR}/cell_pelvis.log
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ${SCRIPT} --summary-only
  "
