#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --job-name=sem50_funccheck
# CPU functional check of the sem50 AugLab config (headline SynthSeg-EM at SynthSeg p 0.5) on a real open-ms FLAIR training batch (TamIA, CPU only).
# See functional_check.py. The trainer is built with nnUNet_results pointed at a throwaway dir (nothing is trained).
#   sbatch --output=<dir>/funccheck_%j.out --export=ALL,OUTDIR=<dir>,N=200 scripts/cluster/sem50/functional_check.sh
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${CHECK_DIR:-${REPO}/scripts/cluster/sem50}"
OUTDIR="${OUTDIR:?}"; mkdir -p "${OUTDIR}"
cd "${REPO}"
source benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh
source scripts/cluster/tamia_env_openms.sh
export nnUNet_results="${OUTDIR}/throwaway_results"; mkdir -p "${nnUNet_results}"
export PYTHONPATH="${SCRIPTS_DIR}:${REPO}:${PYTHONPATH:-}" NNUNET_PROJECT_ROOT="${REPO}"
export nnUNet_n_proc_DA=0 nnUNet_compile=0 WANDB_MODE=disabled NNUNET_NUM_EPOCHS=1 BLOSC2_NTHREADS=1 BLOSC_NTHREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[env] AugLab HEAD $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse --short HEAD); repo HEAD $(git rev-parse --short HEAD); preprocessed ${nnUNet_preprocessed}"
"${REPO}/.venv/bin/python" "${HERE}/functional_check.py" --cfg-dir "${REPO}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" \
    --n "${N:-200}" --out "${OUTDIR}/functional_check.json"
