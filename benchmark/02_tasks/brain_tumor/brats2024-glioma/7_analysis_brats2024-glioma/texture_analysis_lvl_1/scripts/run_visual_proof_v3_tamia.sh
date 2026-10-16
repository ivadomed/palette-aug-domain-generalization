#!/usr/bin/env bash
# TamIA GPU job for visual_proof_v3_activations.py (representation-level measurements on the BraTS 70 test cases, 4 cells, one per GPU;
# whole h100 node because h100 is by-node only; expected << 1 h). Plots afterwards on CPU: run_visual_proof_v3_plots_tamia.sh.
#   bash run_visual_proof_v3_tamia.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_GPU_TYPE=h100 RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
LOGS_DIR="${SCRATCH}/visual_proof_v3"; mkdir -p "${LOGS_DIR}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"
SCRIPT="${A}/scripts/visual_proof_v3_activations.py"; T="benchmark/02_tasks"
run_job --name vp3_activations --gpus 4 --time 01:00:00 \
  --log "${LOGS_DIR}/gpu_$(date +%Y%m%d_%H%M%S).log" -- \
  bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export nnUNet_compile=false
    export PYTHONPATH='${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma:${PROJECT_ROOT}/${T}/brain_ms/open-ms/5_scripts_open-ms:${PROJECT_ROOT}/${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PYTHONPATH:-}'
    export VP3_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_v3_activations'
    export VP_OPENMS_ROOT='${SCRATCH}/open-ms'
    export VP_PELVIS_ROOT='${SCRATCH}/totalseg-pelvic'
    export VP_BRATS_RAW='${SCRATCH}/brats2024-glioma/2_nnUNet/raw/Dataset052_BraTS2024GliomaT2w'
    export VP_BRATS_PRED='${SCRATCH}/brats2024-glioma/8_results/01_predictions/brats2024_glioma_model/t2w/auglab'
    export VP_BRATS_PRED_T1N='${SCRATCH}/brats2024-glioma/8_results/01_predictions/brats2024_glioma_model/t1n/auglab'
    cd '${PROJECT_ROOT}'
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ${SCRIPT} --cells bratsfull_t2w_t1n > ${LOGS_DIR}/cell_t2w_t1n.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ${SCRIPT} --cells bratsfull_t2w_t2f > ${LOGS_DIR}/cell_t2w_t2f.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python ${SCRIPT} --cells bratsfull_t1n_t2f > ${LOGS_DIR}/cell_t1n_t2f.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python ${SCRIPT} --cells bratsfull_t1n_t2w > ${LOGS_DIR}/cell_t1n_t2w.log 2>&1 &
    wait
    tail -n 3 ${LOGS_DIR}/cell_*.log
  "
