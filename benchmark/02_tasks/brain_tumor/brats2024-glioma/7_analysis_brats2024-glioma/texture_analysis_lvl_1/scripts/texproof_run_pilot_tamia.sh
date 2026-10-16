#!/usr/bin/env bash
# texproof pilot on TamIA (2026-10-10, Vulcan's GPU queue was stuck): ONE whole-node H100 job, the two cells in parallel on GPU 0 (Open-MS)
# and GPU 1 (pelvis CT). Run ON TamIA from the repo root:  bash <this file>
# Data = the scratch mirror /scratch/p/paulh/{open-ms,totalseg-pelvic} (same layout below the dataset dir as Vulcan's repo tree).
# TamIA's venv trainer shims are stale (pre-restructuring paths), so the datasets' own 5_scripts dirs go on PYTHONPATH; the venv is not touched.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
REPO="$PWD"; PY="${REPO}/.venv/bin/python"
LVL1="${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"
LOGDIR="${LVL1}/outputs/logs"; mkdir -p "${LOGDIR}"
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_GPU_TYPE=h100 RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
export SCRATCH="${SCRATCH:-/scratch/p/${USER}}"
PP="${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms:${REPO}/benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PYTHONPATH:-}"
run_job --name texproof_tamia --gpus 4 --time 01:00:00 \
    --log "${LOGDIR}/texproof_pilot_tamia.log" -- \
    bash -c "
    export NNUNET_PROJECT_ROOT='${REPO}'
    export PYTHONPATH='${PP}'
    export SCRATCH='${SCRATCH}'
    export TEXPROOF_OPENMS_ROOT='${SCRATCH}/open-ms'
    export TEXPROOF_PELVIS_ROOT='${SCRATCH}/totalseg-pelvic'
    export nnUNet_raw='${nnUNet_raw}'; export nnUNet_preprocessed='${nnUNet_preprocessed}'
    cd '${REPO}'
    nvidia-smi -L
    CUDA_VISIBLE_DEVICES=0 '${PY}' '${LVL1}/scripts/texproof_perturbation_pilot.py' --dataset openms --occlusion patient12 > '${LOGDIR}/texproof_pilot_openms.log' 2>&1 &
    CUDA_VISIBLE_DEVICES=1 '${PY}' '${LVL1}/scripts/texproof_perturbation_pilot.py' --dataset pelvis > '${LOGDIR}/texproof_pilot_pelvis.log' 2>&1 &
    wait
    echo ALL_DONE; tail -n 25 '${LOGDIR}/texproof_pilot_openms.log' '${LOGDIR}/texproof_pilot_pelvis.log'
    "
